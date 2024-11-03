from utils.llm_def import LLaMA


model = LLaMA("/disk/disk_20T/share/Llama-3-8B")
response = model.get_chat_response(
    prompt="你好"
)
# 打印响应
print("模型响应:", response)

# TODO: setting the tokenizer.chat_template attribute
# ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
