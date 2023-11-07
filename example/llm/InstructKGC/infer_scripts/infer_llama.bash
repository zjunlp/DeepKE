CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'models/llama-7b' \
    --model_name 'llama' \
    --lora_weights 'lora/llama-7b-8bit' \
    --input_file 'data/valid.json' \
    --output_file 'results/llama-valid.json' \
    --fp16 \
    --bits 8 
