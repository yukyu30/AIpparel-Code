from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import BitsAndBytesConfig, CLIPVisionModel, LlamaConfig
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from data.datasets.utils import IMAGE_TOKEN_INDEX
from data.garment_tokenizers.special_tokens import PanelEdgeTypeIndices, PanelEdgeType
from .encodings import SinusoidalEncoding, DiscreteEncoding
from data.datasets.panel_configs import StandardizeConfig

def make_mlp(input_dim, hidden_dim, output_dim, num_layers, dropout=0):
    """ Very simple multi-layer perceptron (also called FFN)"""
    h = [input_dim] + [hidden_dim] * (num_layers - 1)
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(h[i], h[i +1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(h[-1], output_dim))
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class Discretize(nn.Module):
    def __init__(self, bin, bounds, dim):
        self.bin = bin 
        self.min_bounds = torch.tensor(bounds[:dim])
        self.max_bounds = torch.tensor(bounds[dim:])
    def forward(self, x):
        x = torch.minimum(self.max_bounds.cuda(), x)
        x = torch.maximum(self.min_bounds.cuda(), x)
        x = (x - self.min_bounds) / (self.max_bounds - self.min_bounds)
        x = torch.round(x * 256) / 256
        x = x * (self.max_bounds - self.min_bounds) + self.min_bounds
        return x
    
def _discretize(x, bin, bounds, dim):
    min_bounds = torch.tensor(bounds[:dim]).cuda()
    max_bounds = torch.tensor(bounds[dim:]).cuda()
    x = torch.minimum(max_bounds.cuda(), x)
    x = torch.maximum(min_bounds.cuda(), x)
    x = (x - min_bounds) / (max_bounds - min_bounds)
    x = torch.round(x * 256) / 256
    x = x * (max_bounds - min_bounds) + min_bounds
    return x
        
        

@dataclass
class GarmentTokenRegressionOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    params: Optional[Dict[int, torch.FloatTensor]] = None
    params_mask_dict: Optional[Dict[int, torch.BoolTensor]] = None
    
class GarmentTokenRegressionMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.config.panel_edge_indices = kwargs["panel_edge_indices"]
        self.config.pos_embed = kwargs.get("pos_embed", False)
        self.config.num_freq = kwargs.get("num_freq", 9)
        self.config.num_regression_layers = kwargs.get("num_regression_layers", 2)
        self.config.pos_embed_type = kwargs.get("pos_embed_type", "sinusoidal")
        self.config.bin_num = kwargs.get("bin_num", 128)
        self.config.discretize = kwargs.get("discretize", False)
        self.config.verts_bounds = kwargs.get("bin_bounds", [-4, -4, 4, 4])
        self.config.transf_bounds = kwargs.get("transf_bounds", [-4, -4, -4, -1, -1, -1, -1, 4, 4, 4, 1, 1, 1, 1])
        self.initialize_panel_edge_modules(self.config)
        
    def initialize_panel_edge_modules(self, config):
        in_dim = config.hidden_size
        # transformation_fc_layers = [
        #     nn.Linear(in_dim, in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_dim, self.config.panel_edge_indices.get_index_param_num(self.config.panel_edge_indices.move_idx)),
        #     nn.Dropout(0.0),
        # ]
        self.transformation_fc = make_mlp(
            in_dim, 
            in_dim,
            self.config.panel_edge_indices.get_index_param_num(self.config.panel_edge_indices.move_idx),
            self.config.num_regression_layers
            )
        self.transformation_fc.train()
        for p in self.transformation_fc.parameters():
            p.requires_grad = True
            
        line_curve_out_dim = 4 if self.config.panel_edge_indices.get_token_indices(PanelEdgeType.CUBIC) == -1 else 6
        if self.config.panel_edge_indices.get_token_indices(PanelEdgeType.ARC) != -1:
            line_curve_out_dim += 2
        
        self.line_curve_fc = make_mlp(
            in_dim, 
            in_dim,
            line_curve_out_dim,
            self.config.num_regression_layers
        )
        self.line_curve_fc.train()
        for p in self.line_curve_fc.parameters():
            p.requires_grad = True
        # if self.config.panel_edge_indices.get_token_indices(PanelEdgeType.ARC) != -1:
        #     arc_fc = nn.Sequential(*[
        #         nn.Linear(in_dim, in_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(in_dim, 4),
        #         nn.Dropout(0.0),
        #     ])
        #     self.arc_fc = arc_fc
        #     self.line_curve_fc.train()
        #     for p in self.line_curve_fc.parameters():
        #         p.requires_grad = True
        # else:
        #     self.arc_fc = None
        if self.config.pos_embed:
            if self.config.pos_embed_type == "sinusoidal":
                self.vertex_encoding = SinusoidalEncoding(
                    in_dim=2,
                    num_frequencies=self.config.num_freq,
                    min_freq_exp=0.0, max_freq_exp=self.config.num_freq - 1, include_input=True
                )
                self.trasl_encoding = SinusoidalEncoding(
                    in_dim=3,
                    num_frequencies=self.config.num_freq,
                    min_freq_exp=0.0, max_freq_exp=self.config.num_freq - 1, include_input=True
                )
                self.transf_proj = nn.Sequential(
                    nn.Linear(self.trasl_encoding.get_out_dim()+4, in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim)
                )
                self.vertex_proj = nn.Sequential(
                    nn.Linear(self.vertex_encoding.get_out_dim(), in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim)
                )
                
            elif self.config.pos_embed_type == "discrete":
                self.vertex_encoding = DiscreteEncoding(
                    in_dim=2,
                    out_dim=in_dim,
                    bin_num=self.config.bin_num,
                    max_bounds = self.config.verts_bounds[2:],
                    min_bounds = self.config.verts_bounds[:2],
                )
                self.trasl_encoding = DiscreteEncoding(
                    in_dim=7,
                    out_dim=in_dim,
                    bin_num=self.config.bin_num,
                    max_bounds = self.config.transf_bounds[7:],
                    min_bounds = self.config.transf_bounds[:7],
                )
                self.transf_proj = nn.Sequential(
                    nn.Linear(self.trasl_encoding.get_out_dim(), in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim)
                )
                self.vertex_proj = nn.Sequential(
                    nn.Linear(self.vertex_encoding.get_out_dim(), in_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim)
                )
        
    def get_trainable_param_names(self):
        l = ["line_curve_fc", "transformation_fc"]
        if self.config.pos_embed:
            l.extend(["vertex_proj", "transf_proj"])
        return l
    
    def init_trainable_params(self):
        self.line_curve_fc[-2].weight.data.zero_()
        # self.arc_fc[-2].weight.data.zero_()
        self.transformation_fc[-2].weight.data.zero_()
        if self.config.pos_embed:
            self.vertex_proj[-1].weight.data.zero_()
            self.transf_proj[-1].weight.data.zero_()
    

def denormalize(gt_stats: StandardizeConfig, params: torch.FloatTensor, is_transf: bool=False):
    shift = gt_stats.translations.shift + gt_stats.rotations.shift if is_transf else gt_stats.vertices.shift
    scale = gt_stats.translations.scale + gt_stats.rotations.scale if is_transf else gt_stats.vertices.scale
    shift = torch.tensor(shift).to(params)
    scale = torch.tensor(scale).to(params)
    if params.ndim == 1:
        params = params.unsqueeze(0)
    assert params.shape[-1] == len(shift) == len(scale)
    params = (params * scale) + shift
    return params
            
            
@dataclass 
class GarmentTokenConfig():
    use_mm_start_end: bool = True


class GarmentTokenRegressionModel(GarmentTokenRegressionMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super(GarmentTokenRegressionModel, self).__init__(config, **kwargs)

        self.config.use_cache = True
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        edge_mask: Optional[torch.BoolTensor] = None,
        transf_mask: Optional[torch.BoolTensor] = None,
        endpoints: Optional[torch.FloatTensor] = None,
        endpoints_mask: Optional[torch.BoolTensor] = None,
        transformations: Optional[torch.FloatTensor] = None,
        transformations_mask: Optional[torch.BoolTensor] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.config.pos_embed:
            if (edge_mask is not None and endpoints is not None and endpoints_mask is not None):
                assert edge_mask.sum() == endpoints_mask.sum(), "edge mask has shape {} but endpoints mask has shape {}" \
                    .format(edge_mask.sum(), endpoints_mask.sum())
                for i in range(inputs_embeds.shape[0]):
                    _endpoints = endpoints[i][endpoints_mask[i]]
                    if self.config.discretize:
                        _endpoints = _discretize(_endpoints, self.config.bin_num, self.config.verts_bounds, 2)
                    edge_embeds = self.vertex_proj(self.vertex_encoding(_endpoints))
                    inputs_embeds[i, edge_mask[i]] = inputs_embeds[i, edge_mask[i]] + edge_embeds
                    
                # endpoints = endpoints[endpoints_mask]
                # edge_embeds = self.vertex_proj(self.vertex_encoding(endpoints))
                # inputs_embeds[edge_mask] = inputs_embeds[edge_mask] + edge_embeds
            if (transf_mask is not None and transformations is not None and transformations_mask is not None):
                assert transf_mask.sum() == transformations_mask.sum()
                for i in range(inputs_embeds.shape[0]):
                    _transformations = transformations[i][transformations_mask[i]]
                    if self.config.discretize:
                        _transformations = _discretize(_transformations, self.config.bin_num, self.config.transf_bounds, 7)
                    transf_embeds = self.transf_proj(torch.cat([self.trasl_encoding(_transformations[:, :3]), _transformations[:, 3:]], dim=1))
                    inputs_embeds[i, transf_mask[i]] = inputs_embeds[i, transf_mask[i]] + transf_embeds
                # transformations = transformations[transformations_mask]
                # if self.config.pos_embed_type == "sinusoidal":
                #     transf_embeds = self.transf_proj(torch.cat([self.trasl_encoding(transformations[:, :3]), transformations[:, 3:]], dim=1))
                # elif self.config.pos_embed_type == "discrete":
                #     transf_embeds = self.transf_proj(self.trasl_encoding(transformations))
                # inputs_embeds[transf_mask] = inputs_embeds[transf_mask] + transf_embeds
            
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return outputs
            
class GarmentTokenRegressionForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        **kwargs,
    ):
        self.panel_edge_indices: PanelEdgeTypeIndices = kwargs["panel_edge_indices"]
        self.gt_stats: Optional[StandardizeConfig] = kwargs.pop("gt_stats", None)
        self.zero_tensor = -torch.tensor(self.gt_stats.vertices.shift) / torch.tensor(self.gt_stats.vertices.scale) if self.gt_stats is not None else torch.zeros(2)
        self.denormalize_for_loss = kwargs.pop("denormalize_for_loss", False)
        self.edge_loss_weight = kwargs.pop("edge_loss_weight", 1.0)
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower
            
        super().__init__(config)

        self.model = GarmentTokenRegressionModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def prepare_edge_transf_masks_for_inputs(
        self, 
        input_ids, 
        return_separate=False,
        pad_for_image=True,
    ):
        image_embeds_len=255
        edge_mask = torch.isin(input_ids, torch.tensor(self.panel_edge_indices.get_all_edge_indices()).to(input_ids))
        transf_mask = input_ids == self.panel_edge_indices.get_token_indices(PanelEdgeType.MOVE)
            
        if IMAGE_TOKEN_INDEX in input_ids and pad_for_image:
            new_edge_mask = torch.zeros([edge_mask.shape[0], edge_mask.shape[1]+image_embeds_len]).bool().cuda()
            new_transf_mask = torch.zeros([transf_mask.shape[0], transf_mask.shape[1]+image_embeds_len]).bool().cuda()
            for i in range(input_ids.shape[0]):
                ## hack for IMAGE_TOKEN_INDEX if there's image in the front: add image_embeds_len (255 for llava, 575 for llava1.5) zeros in the front
                ## why? if there's image in the front, LLaVA will insert the image embeeding there
                if IMAGE_TOKEN_INDEX in input_ids[i]:
                    new_edge_mask[i, image_embeds_len:] = edge_mask[i]
                    new_transf_mask[i, image_embeds_len:] = transf_mask[i]
                # if no image in the front, pad 255 zeros in the end
                # if no image, the pose token remains the same place
                else:
                    new_edge_mask[i, :edge_mask.shape[1]] = edge_mask[i]
                    new_transf_mask[i, :transf_mask.shape[1]] = transf_mask[i]
            edge_mask = new_edge_mask
            transf_mask = new_transf_mask
        if return_separate:
            edge_mask_dict = {}
            for ind in self.panel_edge_indices.get_all_edge_indices():
                mask = input_ids == ind
                if not mask.any():
                    continue
                if IMAGE_TOKEN_INDEX in input_ids and pad_for_image:
                    new_mask = torch.zeros([mask.shape[0], mask.shape[1]+image_embeds_len]).bool().cuda()
                    for i in range(input_ids.shape[0]):
                        ## hack for IMAGE_TOKEN_INDEX if there's image in the front: add image_embeds_len (255 for llava, 575 for llava1.5) zeros in the front
                        ## why? if there's image in the front, LLaVA will insert the image embeeding there
                        if IMAGE_TOKEN_INDEX in input_ids[i]:
                            new_mask[i, image_embeds_len:] = mask[i]
                        # if no image in the front, pad 255 zeros in the end
                        # if no image, the pose token remains the same place
                        else:
                            new_mask[i, :mask.shape[1]] = mask[i]
                    mask = new_mask
                edge_mask_dict[ind] = mask
            return edge_mask, transf_mask, edge_mask_dict
        return edge_mask, transf_mask
            

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None, 
        images=None, 
        last_hidden_state=None,
        edge_mask=None, 
        transf_mask=None, 
        endpoints=None, 
        endpoints_mask=None, 
        transformations=None, 
        transformations_mask=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            last_hidden_state = last_hidden_state[:, -1:, :]
        if self.config.pos_embed and \
            last_hidden_state is not None:
            endpoints, endpoints_mask = None, None 
            transformations, transformations_mask = None, None
            edge_mask, transf_mask, edge_mask_dict = self.prepare_edge_transf_masks_for_inputs(input_ids, return_separate=True, pad_for_image=past_key_values is None)
            if edge_mask.any():
                assert edge_mask.shape[1] == last_hidden_state.shape[1]
                endpoints = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1], 2).to(last_hidden_state)
                endpoints_mask = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1]).bool().cuda()
                for ind, mask in edge_mask_dict.items():
                    if not mask.any():
                        continue
                    endpoints_mask |= mask
                    edge_type = self.panel_edge_indices.get_index_token(ind)
                    edge_embeds = last_hidden_state[mask]
                    if edge_type.is_closure():
                        endpoints[mask] = self.zero_tensor.to(edge_embeds)
                    if edge_type == PanelEdgeType.CUBIC:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeType.ARC:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeType.LINE:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                    elif edge_type == PanelEdgeType.CURVE:
                        endpoints[mask] = self.model.line_curve_fc(edge_embeds)[:, :2]
                        
        
            if transf_mask.any():
                assert transf_mask.shape[1] == last_hidden_state.shape[1]
                transformations = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1], 7).to(last_hidden_state)
                transformations_mask = torch.zeros(last_hidden_state.shape[0], last_hidden_state.shape[1]).bool().cuda()
                transf_embeds = last_hidden_state[transf_mask]
                transformations[transf_mask] = self.model.transformation_fc(transf_embeds)
                transformations_mask[transf_mask] = True
            
            

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "edge_mask": edge_mask,
                "transf_mask": transf_mask,
                "endpoints": endpoints,
                "endpoints_mask": endpoints_mask,
                "transformations": transformations,
                "transformations_mask": transformations_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            standardize_cache_format=standardize_cache_format,
        )
        model_kwargs["last_hidden_state"] = outputs.hidden_states[-1]
        return model_kwargs

    def prepare_edge_panel_mask_dict_for_labels(self, input_ids, labels, image_embeds_len=255):
        edge_panel_mask_dict = {}
        for index in self.panel_edge_indices.get_all_indices():
            token_mask = labels[:, 1:] == index
            token_mask = torch.cat(
                [
                    token_mask,
                    torch.zeros((token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            ## if there exists IMAGE, then pad 255. 
            if IMAGE_TOKEN_INDEX in input_ids:
                new_token_mask = torch.zeros([token_mask.shape[0], token_mask.shape[1]+image_embeds_len]).bool().cuda()
                for i in range(input_ids.shape[0]):
                    ## hack for IMAGE_TOKEN_INDEX if there's image in the front: add image_embeds_len (255 for llava, 575 for llava1.5) zeros in the front
                    ## why? if there's image in the front, LLaVA will insert the image embeeding there
                    if IMAGE_TOKEN_INDEX in input_ids[i]:
                        new_token_mask[i, image_embeds_len:] = token_mask[i]
                    # if no image in the front, pad 255 zeros in the end
                    # if no image, the pose token remains the same place
                    else:
                        new_token_mask[i, :token_mask.shape[1]] = token_mask[i]
                token_mask = new_token_mask
            edge_panel_mask_dict[index] = token_mask
        return edge_panel_mask_dict
        

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
        param_targets: Dict[int, torch.FloatTensor],
        param_target_endpoints: torch.FloatTensor,
        param_target_endpoints_mask: torch.BoolTensor,
        param_target_transformations: torch.FloatTensor,
        param_target_transformations_mask: torch.BoolTensor,
        param_target_masks: Dict[int, torch.BoolTensor],
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        inference: bool = False,
        **kwargs,
    ):
        batch_size = images_clip.shape[0]
        assert batch_size == len(offset) - 1
        #-------------------------- LLaVA part: image + text -> hidden states --------------------------#
        image_embeds_len = 255
        edge_panel_mask_dict = self.prepare_edge_panel_mask_dict_for_labels(input_ids, labels, image_embeds_len)
            
        if self.config.pos_embed:
            edge_mask, transf_mask = self.prepare_edge_transf_masks_for_inputs(input_ids)
        else:
            edge_mask = None
            transf_mask = None

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
            output_hidden_states=True,
            edge_mask=edge_mask,
            transf_mask=transf_mask,
            endpoints=param_target_endpoints,
            endpoints_mask=param_target_endpoints_mask,
            transformations=param_target_transformations,
            transformations_mask=param_target_transformations_mask,
            reduce=False
        )
        
        last_hidden_state = output.hidden_states[-1]
        param_preds = {k:torch.zeros_like(v) for k,v in param_targets.items()}
        edge_type_losses = {}
        total_loss = None
        if param_target_masks is not None:
            total_edge_loss = 0
            for ind in param_target_masks.keys():
                mask = edge_panel_mask_dict[ind]
                if not mask.any():
                    edge_type_losses[f"{self.panel_edge_indices.get_index_token(ind).value}_loss"] = torch.zeros(1).to(last_hidden_state.device)
                    continue
                panel_embeds = last_hidden_state[mask]
                edge_type = self.panel_edge_indices.get_index_token(ind)
                if edge_type == PanelEdgeType.MOVE:
                    panel_params = self.model.transformation_fc(panel_embeds)
                else:
                    panel_params = self.model.line_curve_fc(panel_embeds)
                    if edge_type == PanelEdgeType.CUBIC:
                        panel_params = panel_params[:, :-2]
                    elif edge_type == PanelEdgeType.ARC:
                        panel_params = torch.cat([panel_params[:, :2], panel_params[:, 6:]], dim=-1)
                    elif edge_type == PanelEdgeType.LINE:
                        panel_params = panel_params[:, :2]
                    elif edge_type == PanelEdgeType.CURVE:
                        panel_params = panel_params[:, :4]
                    elif edge_type == PanelEdgeType.CLOSURE_CURVE:
                        panel_params = panel_params[:, 2:4]
                    elif edge_type == PanelEdgeType.CLOSURE_ARC:
                        panel_params = panel_params[:, 6:]
                    elif edge_type == PanelEdgeType.CLOSURE_CUBIC:
                        panel_params = panel_params[:, 2:6]
                    
                param_preds[ind][param_target_masks[ind]] = panel_params 
                if self.gt_stats is not None and self.denormalize_for_loss:
                    param_preds[ind] = denormalize(self.gt_stats, param_preds[ind], is_transf=edge_type == PanelEdgeType.MOVE)
                    param_targets[ind] = denormalize(self.gt_stats, param_targets[ind], is_transf=edge_type == PanelEdgeType.MOVE)
                loss = torch.sum((param_preds[ind] - param_targets[ind]) ** 2, -1).sum(1) / (torch.sum(param_target_masks[ind], 1) + 1e-5)
                total_edge_loss += loss
                edge_type_losses[f"{self.panel_edge_indices.get_index_token(ind).value}_loss"] = loss.mean()
            
            total_loss = self.edge_loss_weight * total_edge_loss
        logits = output.logits
        
        ce_loss = output.loss
        if ce_loss is not None:
            total_loss += ce_loss
        
        
        return_dict = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "edge_loss": total_edge_loss,
            "params": param_preds,
            "logits": logits,
        }
        return_dict.update(edge_type_losses)
        return return_dict

    def evaluate(
        self,
        images_clip,
        input_ids,
        attention_mask,
        endpoints=None, 
        endpoints_mask=None,
        transformations=None,
        transformations_mask=None,
        max_new_tokens=32
    ):
        if self.config.pos_embed:
            edge_mask, transf_mask = self.prepare_edge_transf_masks_for_inputs(input_ids)
        else:
            edge_mask = None
            transf_mask = None
            
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                endpoints=endpoints,
                endpoints_mask=endpoints_mask,
                transformations=transformations,
                transformations_mask=transformations_mask,
                edge_mask=edge_mask,
                transf_mask=transf_mask,
            )
            output_ids = outputs.sequences
            edge_panel_mask_dict = {}
            image_embeds_len = 255
            for ind in self.panel_edge_indices.get_all_indices():
                if self.panel_edge_indices.get_index_param_num(ind) == 0:
                    continue
                token_mask = output_ids[:, 1:] == ind
                token_mask[:,:input_ids.shape[1] - 1] = False
                edge_panel_mask_dict[ind] = token_mask
            
            if IMAGE_TOKEN_INDEX in input_ids:
                for ind, mask in edge_panel_mask_dict.items():
                    new_token_mask = torch.zeros([mask.shape[0], mask.shape[1]+image_embeds_len]).bool().cuda()
                    for i in range(input_ids.shape[0]):
                        if IMAGE_TOKEN_INDEX in input_ids[i]:
                            new_token_mask[i, image_embeds_len:] = mask[i]
                        else:
                            new_token_mask[i, :mask.shape[1]] = mask[i]
                    edge_panel_mask_dict[ind] = new_token_mask
                
            last_hidden_state = torch.cat([outputs.hidden_states[i][-1] for i in range(len(outputs.hidden_states))], dim=1)
            
            param_preds = {}
            for ind, mask in edge_panel_mask_dict.items():
                if mask.sum() == 0:
                    continue
                panel_embeds = last_hidden_state[mask]
                edge_type = self.panel_edge_indices.get_index_token(ind)
                if edge_type == PanelEdgeType.MOVE:
                    panel_params = self.model.transformation_fc(panel_embeds)
                else:
                    panel_params = self.model.line_curve_fc(panel_embeds)
                    if edge_type == PanelEdgeType.CUBIC:
                        panel_params = panel_params[:, :-2]
                    elif edge_type == PanelEdgeType.ARC:
                        panel_params = torch.cat([panel_params[:, :2], panel_params[:, 6:]], dim=-1)
                    elif edge_type == PanelEdgeType.LINE:
                        panel_params = panel_params[:, :2]
                    elif edge_type == PanelEdgeType.CURVE:
                        panel_params = panel_params[:, :4]
                    elif edge_type == PanelEdgeType.CLOSURE_CURVE:
                        panel_params = panel_params[:, 2:4]
                    elif edge_type == PanelEdgeType.CLOSURE_ARC:
                        panel_params = panel_params[:, 6:]
                    elif edge_type == PanelEdgeType.CLOSURE_CUBIC:
                        panel_params = panel_params[:, 2:6]
                        
                param_preds[ind] = panel_params

        return {"output_ids": output_ids, "params": param_preds}
    
