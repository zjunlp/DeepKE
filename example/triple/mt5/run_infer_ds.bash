model_name="output/ccks_mt5-base_f1_1e-4"
output_dir="output/ccks_mt5-base_f1_1e-4_test_result"
data_dir="data"

deepspeed  --include localhost:0 run_finetune.py \
    --do_predict \
    --predict_with_generate \
    --use_fast_tokenizer=True \
    --per_device_eval_batch_size 16 \
    --test_file=${data_dir}/valid.json \
    --model_name_or_path=${model_name}   \
    --output_dir=${output_dir}  \
    --overwrite_output_dir=False \
    --logging_dir=${output_dir}_log \
    --preprocessing_num_workers 4 \
    --generation_max_length 256 \
    --generation_num_beams 1 \
    --gradient_checkpointing=True \
    --bf16=True \
    --deepspeed "configs/ds_mt5_z3_config_bf16.json" \
    --seed 42 