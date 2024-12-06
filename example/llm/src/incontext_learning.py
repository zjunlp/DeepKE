"""
incontext_learning.py

The entrance of this project.
"""

import re
import os
import argparse

from utils.prompt_crafting import PromptCraft
from model.llm_def import LLaMA, Qwen, MiniCPM, ChatGLM, ChatGPT, DeepSeek
from datamodule.preprocess_icl import prepare_examples


MODEL_CLASSES = {
    "LLaMA": LLaMA,
    "Qwen": Qwen,
    "MiniCPM": MiniCPM,
    "ChatGLM": ChatGLM,
    "ChatGPT": ChatGPT,
    "DeepSeek": DeepSeek
}

MODEL_LIST = list(MODEL_CLASSES.keys())

LANGUAGE_LIST = [
    "en",
    "ch"
]

TASK_LIST = [
    "ner",
    "re",
    "ee",
    "rte",
    "da"
]


def parse_args():
    parser = argparse.ArgumentParser(description="In-context learning script")
    parser.add_argument("--engine", type=str, choices=MODEL_LIST, help="Model engine")
    parser.add_argument("--model_id", type=str, help="Model ID")
    parser.add_argument("--api_key", type=str, help="API key")
    parser.add_argument("--base_url", type=str, help="Base URL")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--top_p", type=float, help="Top P")
    parser.add_argument("--max_tokens", type=int, help="Max tokens")
    parser.add_argument("--stop", type=str, help="Stop")
    parser.add_argument("--task", type=str, choices=TASK_LIST, help="Task")
    parser.add_argument("--language", type=str, choices=LANGUAGE_LIST, help="Language")
    parser.add_argument("--in_context", type=bool, help="In-context learning")
    parser.add_argument("--instruction", type=str, help="Instruction")
    parser.add_argument("--data_path", type=str, help="Data path")
    parser.add_argument("--text_input", type=str, help="Text input")
    parser.add_argument("--domain", type=str, help="Domain")
    parser.add_argument("--labels", type=str, help="Labels")
    parser.add_argument("--head_entity", type=str, help="Head entity")
    parser.add_argument("--head_type", type=str, help="Head type")
    parser.add_argument("--tail_entity", type=str, help="Tail entity")
    parser.add_argument("--tail_type", type=str, help="Tail type")
    return parser.parse_args()


def main():

    # <01: Confirm Config>

    args = parse_args()
    cfg = {
        'engine': args.engine,
        'model_id': args.model_id,
        'api_key': args.api_key,
        'base_url': args.base_url,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_tokens': args.max_tokens,
        'stop': args.stop,
        'task': args.task,
        'language': args.language,
        'in_context': args.in_context,
        'instruction': args.instruction,
        'data_path': args.data_path,
        'text_input': args.text_input,
        'domain': args.domain,
        'labels': re.split(r'[，,]', args.labels) if args.labels else None,
        'head_entity': args.head_entity,
        'head_type': args.head_type,
        'tail_entity': args.tail_entity,
        'tail_type': args.tail_type
    }
    # print("All Your Config: ", cfg) # test

    # <02: Checking Config>

    # check if the model is supported
    if cfg['engine'] not in MODEL_LIST:
        raise ValueError("Your model is not supported now, we only support " + ", ".join(MODEL_LIST) + ". Try again please.")

    # check if the model is closed source, if so, the api_key should be provided
    if cfg['engine'] in ["ChatGPT", "DeepSeek"] and not cfg['api_key']:
        cfg['api_key'] = os.getenv('API_KEY')
        if not cfg['api_key']:
            raise ValueError("Your model is closed source, please provide your API key.")

    # check if the task and language are supported
    if cfg['task'] not in TASK_LIST:
        raise ValueError("The task name should be one of ner, re, ee, rte for Information Extraction or da for Data Augmentation. Try again please.")
    if cfg['language'] not in LANGUAGE_LIST:
        raise ValueError("Now we only support language of 'en' (English) and 'ch' (Chinese). Try again please.")

    # check if the text_input is provided
    if not cfg['text_input']:
        raise ValueError("Text Input can't be null")

    # when in_context is True, examples should be prepared, and data_path should be provided
    examples = None
    if cfg['in_context']:
        examples = prepare_examples(cfg['task'], cfg['language'], cfg['data_path'])

    # <03: Build Prompt>

    ie_prompter = PromptCraft(
        task=cfg['task'],
        language=cfg['language'],
        in_context=cfg['in_context'],
        instruction=cfg['instruction'] if "instruction" in cfg else None,  # not recommend to set instruction by yourself
        example=examples,
    )
    prompt = ie_prompter.build_prompt(
        prompt=cfg['text_input'],
        domain=cfg['domain'] if "domain" in cfg else None,
        labels=cfg['labels'] if "labels" in cfg else None,
        head_entity=cfg['head_entity'] if "head_entity" in cfg else None,
        head_type=cfg['head_type'] if "head_type" in cfg else None,
        tail_entity=cfg['tail_entity'] if "tail_entity" in cfg else None,
        tail_type=cfg['tail_type'] if "tail_type" in cfg else None,
    )
    # print("Your final customized-prompt: " + prompt) # test

    # <04: Model Response>

    ModelClass = MODEL_CLASSES[cfg['engine']]

    # Closed Source
    if cfg['engine'] in ["ChatGPT", "DeepSeek"]:
        model = ModelClass(
            model_name=cfg['model_id'],
            api_key=cfg['api_key'],
            base_url=cfg['base_url'] if "base_url" in cfg else None
        )
        response = model.get_chat_response(
            input=prompt,
            temperature=cfg['temperature'] if "temperature" in cfg else 0.1,
            max_tokens=cfg['max_tokens'] if "max_tokens" in cfg else 512,
            stop=cfg['stop'] if "stop" in cfg else None
        )

    # Open Source
    else:
        model = ModelClass(pretrained_model_name_or_path=cfg['model_id'])
        response = model.get_chat_response(
            prompt=prompt,
            temperature=cfg['temperature'] if "temperature" in cfg else 0.1,
            top_p=cfg['top_p'] if "top_p" in cfg else 0.9,
            max_tokens=cfg['max_tokens'] if "max_tokens" in cfg else 512
        )

    # print("Model Response: " + response) # test
    print(response)


if __name__ == "__main__":
    main()