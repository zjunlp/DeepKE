CUDA_VISIBLE_DEVICES="0" python inference.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'lora/llama-7b-e3-r8' \
    --input_file 'data/valid.json' \
    --output_file 'result/llama_7b_e3_r8.json' \
    --load_8bit \