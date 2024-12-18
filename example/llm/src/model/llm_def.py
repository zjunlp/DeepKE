"""
Surpported Models.
Supports:
- Open Source:LLaMA3, Qwen2.5, MiniCPM3, ChatGLM4
- Closed Source: ChatGPT, DeepSeek
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
import openai
import os
from openai import OpenAI

# The inferencing code is taken from the official documentation

class BaseEngine:
    def __init__(self, pretrained_model_name_or_path: str):
        self.name = None
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    def get_chat_response(self, prompt):
        raise NotImplementedError

class LLaMA(BaseEngine):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.name = "llama"
        self.model_id = pretrained_model_name_or_path
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs[0]["generated_text"][-1]['content'].strip()

class Qwen(BaseEngine):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.name = "qwen"
        self.model_id = pretrained_model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 1024):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return response

class MiniCPM(BaseEngine):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.name = "minicpm"
        self.model_id = pretrained_model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.model.device)
        model_outputs = self.model.generate(
            model_inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens
        )
        output_token_ids = [
            model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
        ]
        response = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()

        return response

class ChatGLM(BaseEngine):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.name = "chatglm"
        self.model_id = pretrained_model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True, tokenize=True).to(self.model.device)
        model_outputs = self.model.generate(
            **model_inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens
        )
        model_outputs = model_outputs[:, model_inputs['input_ids'].shape[1]:]
        response = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0].strip()

        return response

class ChatGPT(BaseEngine):
    def __init__(self, model_name: str, api_key: str, base_url=openai.base_url):
        self.name = model_name
        self.model = model_name
        self.base_url = base_url
        if api_key != "":
            self.api_key = api_key
        else:
            self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_response(self, input, temperature=0.1, max_tokens=1024, stop=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": input},
            ],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        )
        return response.choices[0].message.content

class DeepSeek(BaseEngine):
    def __init__(self, model_name: str, api_key: str, base_url="https://api.deepseek.com"):
        self.name = model_name
        self.model = model_name
        self.base_url = base_url
        if api_key != "":
            self.api_key = api_key
        else:
            self.api_key = os.environ["DEEPSEEK_API_KEY"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_response(self, input, temperature=0.1, max_tokens=1024, stop=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": input},
            ],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        )
        return response.choices[0].message.content
