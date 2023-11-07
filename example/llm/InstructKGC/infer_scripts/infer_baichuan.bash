CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'models/Baichuan-7B-Base' \
    --model_name 'baichuan' \
    --lora_weights 'lora/Baichuan-7B-Base-4bit' \
    --input_file 'data/valid.json' \
    --output_file 'results/baichuan-valid.json' \
    --fp16 \
    --bits 4
