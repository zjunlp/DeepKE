python src/inference_vllm.py \
  --stage sft \
  --model_name_or_path 'lora_results/baichuan2-13b-v1/baichuan2-13b-v1' \
  --model_name 'baichuan' \
  --template 'baichuan2' \
  --do_predict \
  --input_file 'data/input.json' \
  --output_file 'results/baichuan2-13b-IEPile-lora_output.json' \
  --output_dir 'lora_results/test' \
  --batch_size 4 \
  --predict_with_generate \
  --max_source_length 1024 \
  --bf16 \
  --max_new_tokens 512

# python src/run.py --mood infer_vllm