export  CUDA_VISIBLE_DEVICES=5,7,8 
export PYTHONPATH=/home/w4756677/garment/AIpparel-Code 

torchrun --standalone --nnodes=1 --nproc_per_node=3 eval_scripts/eval_llava.py \
 experiment.project_name=GPT2 \
 experiment.run_name=aipparel.train \
 gen_only=False eval_val=False eval_train=False \
 --config-name train_garment_llava_regression_w_pos --config-path ../configs