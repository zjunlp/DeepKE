#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

if true; then
type=context-based
bs=4
bl=3e-5
uls=(4e-4)
accum=4
for ul in ${uls[@]}
do
python -u  ../train_bio.py --data_dir ../dataset/gda \
  --max_height 35 \
  --channel_type $type \
  --bert_lr $bl \
  --transformer_type bert \
  --model_name_or_path allenai/scibert_scivocab_cased \
  --train_file train.data \
  --dev_file dev.data \
  --test_file test.data \
  --train_batch_size $bs \
  --test_batch_size $bs \
  --gradient_accumulation_steps $accum \
  --num_labels 1 \
  --learning_rate $ul \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 10.0 \
  --evaluation_steps 400 \
  --seed 66 \
  --num_class 2 \
  --save_path ../checkpoint/gda/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.pt \
  --log_dir ../logs/gda/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.log
done
fi
