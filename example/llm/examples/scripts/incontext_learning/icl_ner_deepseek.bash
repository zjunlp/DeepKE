python src/incontext_learning.py \
    --engine "DeepSeek" \
    --model_id "deepseek-7b-chat" \
    --api_key "your-api-key" \
    --task "ner" \
    --language "ch" \
    --in_context true \
    --text_input "比尔·盖茨是美国企业家、软件工程师、慈善家、微软公司创始人、中国工程院外籍院士。曾任微软董事长、CEO和首席软件设计师。" \
    --domain "人物" \
    --labels "头衔，任职"