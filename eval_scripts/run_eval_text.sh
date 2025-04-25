#!/bin/bash
export  CUDA_VISIBLE_DEVICES=5,7,8,9
export PYTHONPATH=/home/w4756677/garment/AIpparel-Code 

torchrun --standalone --nnodes=1 --nproc_per_node=4 eval_scripts/eval_llava.py \
 experiment.project_name=AIpparel \
 experiment.run_name=eval_text \
 'data_wrapper.dataset.sampling_rate=[0,1,0,0,0]' \
 pre_trained=/miele/george/garment/aipparel_pretrained.pth \
 gen_only=True eval_val=True eval_train=False \
 --config-name train_garment_llava_regression_w_pos --config-path ../configs