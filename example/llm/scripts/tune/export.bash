python src/export_model.py \
    --model_name_or_path 'models/llama-3-8b-Instruct' \
    --checkpoint_dir 'lora_results/llama3-v1/checkpoint-xxx' \
    --export_dir 'lora_results/llama3-v1/llama3-v1' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama3' \
    --output_dir 'lora_results/test'

# python src/export_model.py --config examples/infer/export.yaml
