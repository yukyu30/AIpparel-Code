export  CUDA_VISIBLE_DEVICES=5,7,8 
export PYTHONPATH=/home/w4756677/garment/AIpparel-Code 

torchrun --standalone --nnodes=1 --nproc_per_node=3 eval_scripts/eval_llava.py \
 experiment.run_name=aipparel.eval \
 pre_trained=/miele/george/garment/sewformer/outputs/2024-10-23/15-56-57/ckpt_16 \
 gen_only=False eval_val=False eval_train=False \
 --config-name train_garment_llava_regression_w_pos --config-path ../configs