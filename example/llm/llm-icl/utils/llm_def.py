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

class BaseModel:
    def __init__(self, pretrained_model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    def get_chat_response(self, prompt):
        raise NotImplementedError

class LLaMA(BaseModel):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
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

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 512):
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


class Qwen(BaseModel):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.model_id = pretrained_model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 512):
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


class MiniCPM(BaseModel):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.model_id = pretrained_model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 512):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
            self.model.device)
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


class ChatGLM(BaseModel):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.model_id = pretrained_model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    def get_chat_response(self, prompt, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 512):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True,
                                                          add_generation_prompt=True, tokenize=True).to(
            self.model.device)
        model_outputs = self.model.generate(
            **model_inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens
        )
        model_outputs = model_outputs[:, model_inputs['input_ids'].shape[1]:]
        response = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0].strip()

        return response


class ChatGPT(BaseModel):
    def __init__(self, model_name: str, api_key: str, base_url=openai.base_url):
        self.model = model_name
        self.base_url = base_url
        if api_key != "":
            self.api_key = api_key
        else:
            self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_response(self, input, temperature=0.1, max_tokens=512, stop=None):
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


class DeepSeek(BaseModel):
    def __init__(self, model_name: str, api_key: str, base_url="https://api.deepseek.com"):
        self.model = model_name
        self.base_url = base_url
        if api_key != "":
            self.api_key = api_key
        else:
            self.api_key = os.environ["DEEPSEEK_API_KEY"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_response(self, input, temperature=0.1, max_tokens=512, stop=None):
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


if __name__ == "__main__":
    test_prompt = "What is the importance of renewable energy?"

    def test_llama():
        print("\n=== Testing LLaMA Model ===")
        try:
            llama_model = LLaMA(pretrained_model_name_or_path="/disk/disk_20T/share/Llama-3-8B")
            llama_response = llama_model.get_chat_response(
                prompt=test_prompt,
                temperature=0.5,
                top_p=0.9,
                max_tokens=100
            )
            print("LLaMA Response:", llama_response)
        except Exception as e:
            print("LLaMA Model Test Failed:", str(e))

    def test_qwen():
        print("\n=== Testing Qwen Model ===")
        try:
            qwen_model = Qwen(pretrained_model_name_or_path="huggingface/Qwen2.5")
            qwen_response = qwen_model.get_chat_response(
                prompt=test_prompt,
                temperature=0.5,
                top_p=0.9,
                max_tokens=100
            )
            print("Qwen Response:", qwen_response)
        except Exception as e:
            print("Qwen Model Test Failed:", str(e))

    def test_minicpm():
        print("\n=== Testing MiniCPM Model ===")
        try:
            minicpm_model = MiniCPM(pretrained_model_name_or_path="huggingface/MiniCPM3")
            minicpm_response = minicpm_model.get_chat_response(
                prompt=test_prompt,
                temperature=0.5,
                top_p=0.9,
                max_tokens=100
            )
            print("MiniCPM Response:", minicpm_response)
        except Exception as e:
            print("MiniCPM Model Test Failed:", str(e))

    def test_chatglm():
        print("\n=== Testing ChatGLM Model ===")
        try:
            chatglm_model = ChatGLM(pretrained_model_name_or_path="huggingface/ChatGLM4")
            chatglm_response = chatglm_model.get_chat_response(
                prompt=test_prompt,
                temperature=0.5,
                top_p=0.9,
                max_tokens=100
            )
            print("ChatGLM Response:", chatglm_response)
        except Exception as e:
            print("ChatGLM Model Test Failed:", str(e))

    def test_chatgpt():
        print("\n=== Testing ChatGPT Model ===")
        try:
            chatgpt_model = ChatGPT(model_name="gpt-3.5-turbo", api_key="your_openai_api_key")
            chatgpt_response = chatgpt_model.get_chat_response(
                input=test_prompt,
                temperature=0.5,
                max_tokens=100
            )
            print("ChatGPT Response:", chatgpt_response)
        except Exception as e:
            print("ChatGPT Model Test Failed:", str(e))

    def test_deepseek():
        print("\n=== Testing DeepSeek Model ===")
        try:
            deepseek_model = DeepSeek(model_name="deepseek-v1", api_key="your_deepseek_api_key")
            deepseek_response = deepseek_model.get_chat_response(
                input=test_prompt,
                temperature=0.5,
                max_tokens=100
            )
            print("DeepSeek Response:", deepseek_response)
        except Exception as e:
            print("DeepSeek Model Test Failed:", str(e))

    # test
    test_llama()
    # test_qwen()
    # test_minicpm()
    # test_chatglm()
    # test_chatgpt()
    # test_deepseek()
