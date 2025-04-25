#!/bin/bash
export  CUDA_VISIBLE_DEVICES=5,7,8 
export PYTHONPATH=/home/w4756677/garment/AIpparel-Code 

torchrun --standalone --nnodes=1 --nproc_per_node=3 eval_scripts/eval_llava.py \
 experiment.project_name=GPT2 \
 experiment.run_name=aipparel.eval \
 'data_wrapper.dataset.sampling_rate=[0,0,0,1,0]' \
 pre_trained=/miele/george/garment/aipparel_pretrained.pth \
 gen_only=True eval_val=False eval_train=True \
 --config-name train_garment_llava_regression_w_pos --config-path ../configs