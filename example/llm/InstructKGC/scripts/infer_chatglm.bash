CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'models/chatglm-6b' \
    --model_name 'chatglm' \
    --lora_weights 'lora/chatglm-6b' \
    --input_file 'data/valid.json' \
    --output_file 'results/chatglm-valid.json' \
    --fp16 
