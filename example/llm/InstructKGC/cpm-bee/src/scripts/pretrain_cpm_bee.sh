#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12345
MODEL_STEPS="0"

OPTS=""
OPTS+=" --model-config config/cpm-bee-10b.json"
OPTS+=" --dataset=datasets.json"
OPTS+=" --batch-size 8"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name cpm_live_checkpoint"
OPTS+=" --max-length 2048"
OPTS+=" --save results/"
OPTS+=" --lr 0.1"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step ${MODEL_STEPS}"
OPTS+=" --log-dir logs/train/"
OPTS+=" --tensorboard logs/tensorboard/cpm_live_48_4096/"
OPTS+=" --load results/cpm_live_checkpoint-${MODEL_STEPS}.pt"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain_cpm_bee.py ${OPTS}"

echo ${CMD}
$CMD

