export  CUDA_VISIBLE_DEVICES=5,7,8 
export PYTHONPATH=/home/w4756677/garment/AIpparel-Code 

torchrun --standalone --nnodes=1 --nproc_per_node=3 scripts/run.py \
 experiment.project_name=AIpparel \
 experiment.run_name=train \
 evaluate=False \
 --config-name aipparel --config-path ../configs