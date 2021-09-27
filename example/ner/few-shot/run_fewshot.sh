#!/usr/bin/env bash

DATASET_NAME=$1 # mit-movie, mit-restaurant, atis
LOAD_PROMPT=$2
BART_NAME="/data/lilei/project/BARTNER-AMAX/facebook/"

if [ $LOAD_PROMPT = "True" ];then
        python -u run.py \
                --dataset_name=${DATASET_NAME} \
                --bart_name=${BART_NAME} \
                --num_epochs=30 \
                --batch_size=16 \
                --learning_rate=1e-4 \
                --warmup_ratio=0.01 \
                --eval_begin_epoch=16 \
                --seed=1 \
                --src_seq_ratio=0.8 \
                --prompt_len=10 \
                --prompt_dim=800 \
                --load_path="save_models/conll2003_16_2e-05/best_model.pth" \
                --freeze_plm \
                --use_prompt \
                --learn_weights \
                --do_train
elif [ $LOAD_PROMPT = "False" ];then
        python -u run.py \
                --dataset_name=${DATASET_NAME} \
                --bart_name=${BART_NAME} \
                --num_epochs=30 \
                --batch_size=16 \
                --learning_rate=1e-4 \
                --warmup_ratio=0.01 \
                --eval_begin_epoch=16 \
                --seed=1 \
                --src_seq_ratio=0.8 \
                --prompt_len=10 \
                --prompt_dim=800 \
                --freeze_plm \
                --use_prompt \
                --learn_weights \
                --do_train
fi


# mit-movie: bsz-16, lr-1e-4
# atis: bsz-4, lr-1e-4