#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/run.py \
 experiment.project_name=AIpparel \
 experiment.run_name=eval_image_text \
 'data_wrapper.dataset.sampling_rate=[0,0,0,1,0]' \
 pre_trained=/miele/george/garment/aipparel_pretrained.pth \
 evaluate=True \
 --config-name aipparel --config-path ../configs