python src/incontext_learning.py \
    --engine "LLaMA" \
    --model_id "damo/Llama-3-8B" \
    --task "rte" \
    --language "ch" \
    --in_context True \
    --text_input "卢浮宫始建于1204年，位于法国巴黎市中心的塞纳河北岸。" \
    --domain "地理"