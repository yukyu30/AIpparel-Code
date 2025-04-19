from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import BitsAndBytesConfig, CLIPVisionModel, LlamaConfig

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)




class GarmentTokenMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GarmentTokenMetaModel, self).__init__(config)

        self.config = config


@dataclass 
class GarmentTokenConfig():
    use_mm_start_end: bool = True
    edge_loss_weight: float = 1.0


class GarmentTokenModel(LlavaLlamaModel):
    def __init__(
        self,
        config,
    ):
        super(GarmentTokenModel, self).__init__(config)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class GarmentTokenForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower
            
        super().__init__(config)

        self.model = GarmentTokenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        question_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        inference: bool = False,
        **kwargs,
    ):
        
        batch_size = images_clip.shape[0]
        assert batch_size == len(offset) - 1



        images_clip_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = (
                images_clip[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_clip_list.append(images_clip_i)
        images_clip = torch.cat(images_clip_list, dim=0)

        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=False,
        )

        logits = output.logits

        loss = output.loss
        
        return {"total_loss":loss, "ce_loss": loss, "logits":logits}

    def evaluate(
        self,
        images_clip,
        input_ids,
        attention_mask,
        max_new_tokens=32
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences


        return {"output_ids": output_ids}
    
