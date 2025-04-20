import os
import sys
from dataclasses import dataclass, field
import logging 
log = logging.getLogger(__name__)
import hydra
from typing import Literal
import torch
import transformers
from typing import Optional
from omegaconf import OmegaConf

from models.garment_token_regression import GarmentTokenRegressionForCausalLM, GarmentTokenConfig
from models.llava import conversation as conversation_lib
from data.data_wrappers.garment_image_data_wrapper import GarmentImageDataWrapper, GarmentImageDataWrapperConfig
from trainers.llava_trainer import FinetuneLlavaTrainer, FinetuneLlavaTrainerConfig, ExperimentConfig
from data.datasets.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN)


@dataclass
class MainConfig:
    version: str
    model_max_length: int
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    trainer: FinetuneLlavaTrainerConfig = field(default_factory=FinetuneLlavaTrainerConfig)
    data_wrapper: GarmentImageDataWrapperConfig = field(default_factory=GarmentImageDataWrapperConfig)
    model: GarmentTokenConfig = field(default_factory=GarmentTokenConfig)
    precision: Literal["bf16", "fp16"] = "bf16"
    eval_only: bool = False
    gen_only: bool = False
    conv_type: Literal["default", "v0", "v1", "vicuna_v1", "llama_2", "plain", "v0_plain", "llava_v0", "v0_mmtag", "llava_v1", "v1_mmtag", "llava_llama_2", "mpt"] = "llava_v1"
    pre_trained: Optional[str] = None
    from_start: bool = False

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory : {output_dir}")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = (ddp_rank == 0)
    

    data_wrapper: GarmentImageDataWrapper = hydra.utils.instantiate(
        cfg.data_wrapper,
        output_dir=output_dir
    )
    log.info("Dataset created.")

    trainer: FinetuneLlavaTrainer = hydra.utils.instantiate(
        cfg.trainer,
        experiment_cfg=cfg.experiment,
        data_wrapper=data_wrapper,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        ddp_local_rank=ddp_local_rank,
        precision=cfg.precision,
    )
    
    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.version,
        cache_dir=None,
        model_max_length=cfg.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    tokenizer.pad_token = tokenizer.unk_token
    all_new_tokens = data_wrapper.dataset.garment_tokenizer.get_all_token_names()
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    if master_process:
        log.info(f"Added {num_added_tokens} tokens to the tokenizer.")
    token_name2_idx_dict = {}
    for token in all_new_tokens:
        token_idx = tokenizer(token, add_special_tokens=False).input_ids[0]
        token_name2_idx_dict[token] = token_idx
        
    if master_process:
        log.info(f"Token name to index dictionary: {token_name2_idx_dict}")
    data_wrapper.dataset.garment_tokenizer.set_token_indices(token_name2_idx_dict)

    if cfg.model.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    torch_dtype = torch.float32
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half
        
    model = GarmentTokenRegressionForCausalLM.from_pretrained(
        cfg.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **cfg.model, vision_tower=data_wrapper.dataset.vision_tower, 
        panel_edge_indices=data_wrapper.dataset.garment_tokenizer.panel_edge_type_indices, gt_stats=data_wrapper.dataset.garment_tokenizer.gt_stats
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=ddp_local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]

    model.resize_token_embeddings(len(tokenizer))

    for name, module in model.named_parameters():
        print(name, module.shape, module.requires_grad)
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    trainer.training_setup(config_dict, model, tokenizer, cfg.conv_type, cfg.from_start, cfg.pre_trained)
    
    if cfg.eval_only:
        trainer.eval_step(trainer.start_epoch * trainer.steps_per_epoch)
        trainer.generation_step(trainer.start_epoch * trainer.steps_per_epoch)
    elif cfg.gen_only:
        trainer.generation_step(trainer.start_epoch)
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
