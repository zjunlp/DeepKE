CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'models/moss-moon-003-sft' \
    --model_name 'moss' \
    --prompt_template_name 'moss' \
    --lora_weights 'lora/moss-4bit' \
    --input_file 'data/valid.json' \
    --output_file 'results/moss-valid.json' \
    --fp16 \
    --bits 4 
