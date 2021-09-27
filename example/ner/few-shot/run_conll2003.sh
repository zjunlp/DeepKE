#!/usr/bin/env bash

DATASET_NAME="conll2003"
BART_NAME="/data/lilei/project/BARTNER-AMAX/facebook/"

python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bart_name=${BART_NAME} \
        --num_epochs=30 \
        --batch_size=16 \
        --learning_rate=2e-5 \
        --warmup_ratio=0.01 \
        --eval_begin_epoch=16 \
        --seed=1 \
        --src_seq_ratio=0.6 \
        --prompt_len=10 \
        --prompt_dim=800 \
        --freeze_plm \
        --use_prompt \
        --learn_weight \
        --do_train