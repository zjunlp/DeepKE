python src/incontext_learning_plus.py \
    --engine "ChatGPT" \
    --model_id "gpt-3.5-turbo" \
    --api_key "your_api_key" \
    --max_tokens 400 \
    --task "re" \
    --language "ch" \
    --in_context true \
    --text_input "../data/RE/try.json"

# python src/incontext_learning_plus.py --config examples/incontext_learning/re_chatgpt_plus.yaml
