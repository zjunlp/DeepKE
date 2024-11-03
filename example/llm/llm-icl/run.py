"""
run.py

<01: Confirm Config>
<02: Build Prompt>
<03: Model Response>
<04: Log and Output>
"""

import hydra
# import logging
import os

from utils.custom_prompt import CustomPrompt
from utils.llm_def import LLaMA, Qwen, MiniCPM, ChatGLM, ChatGPT, DeepSeek
from utils.preprocess import prepare_examples
from utils.set_config import config_from_keyboard


MODEL_CLASSES = {
    "LLaMA": LLaMA,
    "Qwen": Qwen,
    "MiniCPM": MiniCPM,
    "ChatGLM": ChatGLM,
    "ChatGPT": ChatGPT,
    "DeepSeek": DeepSeek
}

# logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    """
    need <config.yaml> be like:
    """
    # <01: Confirm Config>
    # Support Keyboard Input
    # use_keyboard = input("是否要使用键盘输入新的 config 字段？(y/n): ").strip().lower()
    # if use_keyboard == 'y':
    #     config_from_keyboard(cfg)
    #     with open("config.yaml", "r") as yaml_file:
    #         cfg = yaml.safe_load(yaml_file)
    # cwd:
    cfg.cwd = hydra.utils.get_original_cwd()
    # engine:
    if cfg.engine not in MODEL_CLASSES:
        raise ValueError("Your model is not support now.")
    # api_key:
    api_key = cfg.api_key if "api_key" in cfg else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Need an API Key.")
    # task:
    if cfg.task not in ["ner", "re", "ee", "rte", "da"]:
        raise ValueError("The task name should be one of ner, re, ee, rte for Information Extraction or da for Data Augmentation.")
    # language:
    if cfg.language not in ["en", "ch"]:
        raise ValueError("Now we only support language of 'en' (English) and 'ch' (Chinese).")
    # in_context: instruction: / data_path:->examples
    examples = None
    if cfg.in_context:
        examples = prepare_examples(cfg.task, cfg.language, cfg.data_path)
    # text_input:
    if "text_input" not in cfg:
        raise ValueError("Need your text input first!")

    # <02: Build Prompt>
    ie_prompter = CustomPrompt(
        task=cfg.task,
        language=cfg.language,
        in_context=cfg.in_context,
        instruction=cfg.instruction if "instruction" in cfg else None,
        example=examples
    )
    prompt = ie_prompter.build_prompt(
        prompt=cfg.text_input,
        domain=cfg.domain if "domain" in cfg else None,
        labels=cfg.labels if "labels" in cfg else None,
        head_entity=cfg.head_entity if "head_entity" in cfg else None,
        head_type=cfg.head_type if "head_type" in cfg else None,
        tail_entity=cfg.tail_entity if "tail_entity" in cfg else None,
        tail_type=cfg.tail_type if "tail_type" in cfg else None,
    )
    print("Your final customized-prompt: " + prompt)

    # return 1

    # <03: Model Response>
    ModelClass = MODEL_CLASSES[cfg.engine]
    # Closed Source
    if cfg.engine in ["ChatGPT", "DeepSeek"]:
        model = ModelClass(
            model_name=cfg.model_id,
            api_key=api_key,
            base_url=cfg.base_url if "base_url" in cfg else None
        )
        response = model.get_chat_response(
            input=prompt,
            temperature=cfg.temperature if "temperature" in cfg else 0.1,
            max_tokens=cfg.max_tokens if "max_tokens" in cfg else 512,
            stop=cfg.stop if "stop" in cfg else None
        )
    # Open Source
    else:
        model = ModelClass(pretrained_model_name_or_path=cfg.model_id)
        response = model.get_chat_response(
            prompt=prompt,
            temperature=cfg.temperature if "temperature" in cfg else 0.1,
            top_p=cfg.top_p if "top_p" in cfg else 0.9,
            max_tokens=cfg.max_tokens if "max_tokens" in cfg else 512
        )

    # <04: Log and Output>
    print("Model response: " + response)
    # logger.info(response)


if __name__ == '__main__':
    main()
