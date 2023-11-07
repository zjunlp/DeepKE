CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'models/vicuna-7b' \
    --model_name 'vicuna' \
    --prompt_template_name 'vicuna' \
    --lora_weights 'lora/vicuna-7b-8bit' \
    --input_file 'data/valid.json' \
    --output_file 'results/vicuna-valid.json' \
    --fp16 \
    --bits 8 
