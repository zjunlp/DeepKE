"""
cd DeepKE
python -m example.llm.test.test_local_models
"""

from example.llm.src.model.llm_def import LLaMA


model = LLaMA("damo/Llama-3-8B")
# model.pipeline.tokenizer.chat_template = """
# System: You are a helpful assistant.
# User: {user_input}
# Assistant: {assistant_response}
# """
response = model.get_chat_response(
    prompt="How you doing?",
)
print("Re:", response)


# class LLaMA:
#     def __init__(self, pretrained_model_name_or_path):
#         self.model_id = pretrained_model_name_or_path
#         self.pipeline = pipeline(
#             "text-generation",
#             model=self.model_id,
#             model_kwargs={"torch_dtype": torch.bfloat16},
#             device_map="auto",
#         )
#         self.pipeline.tokenizer.chat_template = """
#         System: You are a helpful assistant.
#         User: {user_input}
#         Assistant: {assistant_response}
#         """
#         self.terminators = [
#             self.pipeline.tokenizer.eos_token_id,
#             self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#         ]

#     def get_chat_response(self, prompt, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 150, do_sample: bool = True):
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt},
#         ]
#         outputs = self.pipeline(
#             prompt,
#             temperature=temperature,
#             top_p=top_p,
#             max_new_tokens=max_tokens,
#             return_full_text=False,
#             do_sample=do_sample
#         )
#         return outputs[0]['generated_text']
