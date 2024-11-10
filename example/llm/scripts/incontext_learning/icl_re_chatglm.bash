python src/incontext_learning.py \
    --engine "ChatGLM" \
    --model_id "THUDM/chatglm-6b" \
    --max_tokens 300 \
    --task "ee" \
    --language "ch" \
    --in_context true \
    --text_input "2007年11月6日，阿里巴巴正式以港币13.5元在香港联合交易所挂牌上市，股票代码为“1688 HK”。阿里巴巴上市开盘价30港元，较发行价提高122%。融资116亿港元，创下中国互联网公司融资规模之最。" \
    --domain "财经"