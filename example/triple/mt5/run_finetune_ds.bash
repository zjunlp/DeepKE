model_name="google/mt5-base"
output_dir="output/ccks_mt5-base_f1_1e-4"
data_dir="data"
batch_size=16

deepspeed  --include localhost:0,1 run_finetune.py \
    --do_train --do_eval --do_predict \
    --predict_with_generate \
    --model_name_or_path=${model_name}   \
    --output_dir=${output_dir}  \
    --overwrite_output_dir=False \
    --logging_dir=${output_dir}_log \
    --train_file=${data_dir}/train.json \
    --test_file=${data_dir}/valid.json \
    --use_fast_tokenizer=True \
    --from_checkpoint=True \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --metric_for_best_model "overall-score" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --num_train_epochs 10 \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 3)) \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 4 \
    --generation_max_length 256 \
    --generation_num_beams 1 \
    --gradient_checkpointing=True \
    --bf16=True \
    --deepspeed "configs/ds_mt5_z3_config_bf16.json" \
    --seed 42 
