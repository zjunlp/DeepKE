#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12345

OPTS=""
OPTS+=" --use-delta"
OPTS+=" --model-config config/cpm-bee-10b.json"
OPTS+=" --dataset path/to/dataset"
OPTS+=" --eval_dataset path/to/eval/dataset"
OPTS+=" --epoch 100"
OPTS+=" --batch-size 5"
OPTS+=" --train-iters 100"
OPTS+=" --save-name cpm_bee_finetune"
OPTS+=" --max-length 2048"
OPTS+=" --save results/"
OPTS+=" --lr 0.0001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 1"
OPTS+=" --eval-interval 1000"
OPTS+=" --early-stop-patience 5"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 32768"
OPTS+=" --start-step 0"
OPTS+=" --load model.pt"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} finetune_cpm_bee.py ${OPTS}"

echo ${CMD}
$CMD

