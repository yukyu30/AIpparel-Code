import os
from dataclasses import dataclass, field
import logging 
log = logging.getLogger(__name__)
import hydra
from typing import Literal
import torch
import transformers
from typing import Optional
from torch.utils.data import DataLoader
from functools import partial
import torch.distributed as dist
from PIL import Image

from models.aipparel_model import AIpparelForCausalLM, AIpparelConfig
from models.llava import conversation as conversation_lib
from data.data_wrappers.collate_fns import collate_fn
from data.datasets.inference_dataset import InferenceDataset
from data.datasets.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN)
from trainers.utils import dict_to_cuda, dict_to_cpu, dict_to_dtype


@dataclass
class MainConfig:
    version: str
    model_max_length: int
    model: AIpparelConfig = field(default_factory=AIpparelConfig)
    precision: Literal["bf16", "fp16"] = "bf16"
    conv_type: Literal["default", "v0", "v1", "vicuna_v1", "llama_2", "plain", "v0_plain", "llava_v0", "v0_mmtag", "llava_v1", "v1_mmtag", "llava_llama_2", "mpt"] = "llava_v1"
    pre_trained: Optional[str] = None
    inference_json: str = "assets/data_configs/inference_example.json"
    vision_tower: str = "openai/clip-vit-large-patch14"
    panel_classification: str = "assets/data_configs/panel_classes_garmentcodedata.json"
    garment_tokenizer: str = "gcd_garment_tokenizer"

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory : {output_dir}")


    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.version,
        cache_dir=None,
        model_max_length=cfg.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    dataset = InferenceDataset(
        inference_json=cfg.inference_json,
        vision_tower=cfg.vision_tower,
        image_size=224,
        garment_tokenizer=hydra.utils.instantiate(cfg.garment_tokenizer),
        panel_classification=cfg.panel_classification
    )
    tokenizer.pad_token = tokenizer.unk_token
    all_new_tokens = dataset.get_all_token_names()
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    log.info(f"Added {num_added_tokens} tokens to the tokenizer.")
    token_name2_idx_dict = {}
    for token in all_new_tokens:
        token_idx = tokenizer(token, add_special_tokens=False).input_ids[0]
        token_name2_idx_dict[token] = token_idx
        
    log.info(f"Token name to index dictionary: {token_name2_idx_dict}")
    dataset.set_token_indices(token_name2_idx_dict)

    if cfg.model.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    torch_dtype = torch.float32
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half
        
    model = AIpparelForCausalLM.from_pretrained(
        cfg.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **cfg.model, vision_tower=cfg.vision_tower, 
        panel_edge_indices=dataset.panel_edge_type_indices, gt_stats=dataset.gt_stats
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device="cuda")
    model.resize_token_embeddings(len(tokenizer))

    for p in model.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]


        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    assert cfg.pre_trained is not None
    state_dict = torch.load(cfg.pre_trained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cuda")
    model.eval()
    
    
    
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=cfg.conv_type,
            use_mm_start_end=cfg.model.use_mm_start_end,
            local_rank=0,
            generation_only=True,
        )
    )

    save_path = f'{output_dir}/inference'
    os.makedirs(save_path, exist_ok=True)
    for i, input_dict in enumerate(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        input_dict = dict_to_dtype(input_dict,
            torch_dtype,
            [
                "images_clip",
                "param_targets",
                "param_target_endpoints",
                "param_target_transformations",
                "questions_pattern_endpoints",
                "questions_pattern_transformations"
                
            ]
        )

        output_dict = model.evaluate(
            input_dict["images_clip"],
            input_dict["question_ids"],
            input_dict["question_attention_masks"],
            endpoints=input_dict["questions_pattern_endpoints"],
            endpoints_mask=input_dict["questions_pattern_endpoints_mask"],
            transformations=input_dict["questions_pattern_transformations"],
            transformations_mask=input_dict["questions_pattern_transformations_mask"],
            max_new_tokens=2100
        )
        output_dict = dict_to_cpu(output_dict)
        output_dict = dict_to_dtype(output_dict, torch.float32)
        output_dict["input_mask"] = torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1) >= input_dict["question_ids"].shape[1]
        output_text, patterns, _ = dataset.decode(output_dict, tokenizer)
        try:
            data_name = f"sample_{i}"
            os.makedirs(os.path.join(save_path, data_name), exist_ok=True)
            patterns.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_pred')
            if "gt_patterns" in input_dict:
                for gt_pattern in input_dict["gt_patterns"][0]:
                    gt_pattern.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_gt')
            f = open(os.path.join(save_path, data_name, "output.txt"), "w")
            question = input_dict["questions_list"][0]
            f.write(f"Question: {question}\n")
            f.write(f"Output Text: {output_text}\n")
            f.close()
            if os.path.isfile(input_dict["image_paths"][0]):
                cond_img = Image.open(input_dict["image_paths"][0])
                cond_img.save(os.path.join(save_path, data_name, 'input.png'))
        except Exception as e:
            log.error(e)
            pass


if __name__ == "__main__":
    main()
