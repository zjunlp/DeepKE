#!/usr/bin/env bash

DATASET_NAME="conll2003"
BART_NAME="/data/lilei/project/BARTNER-AMAX/facebook/"

python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bart_name=${BART_NAME} \
        --seed=1 \
        --src_seq_ratio=0.6 \
        --prompt_len=10 \
        --prompt_dim=800 \
        --freeze_plm \
        --use_prompt \
        --learn_weight \
        --load_path="save_models/conll2003_16_2e-05/best_model.pth" \
        --write_path="data/conll2003/predict.txt" \
        --do_test