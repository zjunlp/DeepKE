CUDA_VISIBLE_DEVICES="2" python src/inference.py \
    --model_name_or_path 'models/cpm-bee-10b' \
    --model_name 'cpm-bee' \
    --input_file 'data/valid.json' \
    --lora_weights 'lora/chatglm-6b' \
    --output_file 'results/cpm-bee-5b-valid.json' \
    --fp16 
