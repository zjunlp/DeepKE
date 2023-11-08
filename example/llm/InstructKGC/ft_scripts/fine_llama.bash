output_dir='lora/llama-7b-8bit'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --master_port=1331 src/finetune.py \
    --do_train --do_eval \
    --model_name_or_path 'models/llama-7b' \
    --model_name 'llama' \
    --train_file 'data/train.json' \
    --output_dir=${output_dir}  \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --optim "adamw_torch" \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_memory_MB 24000 \
    --fp16 \
    --bits 4 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err

