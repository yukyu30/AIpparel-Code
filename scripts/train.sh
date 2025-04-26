torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/run.py \
 experiment.project_name=AIpparel \
 experiment.run_name=train \
 evaluate=False \
 --config-name aipparel --config-path ../configs