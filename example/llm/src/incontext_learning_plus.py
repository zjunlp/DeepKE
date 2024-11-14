"""
incontext_learning.py

The entrance of this project.
"""

import hydra
import os

from utils.prompt_crafting import PromptCraft
from model.llm_def import LLaMA, Qwen, MiniCPM, ChatGLM, ChatGPT, DeepSeek
from datamodule.preprocess_icl import prepare_examples, prepare_input_plus


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


@hydra.main(version_base=None, config_path="../examples/incontext_learning", config_name="re_chatgpt_plus")
def main(cfg):
    # <Checking Config>
    # rebuild text_input
    cfg.text_input = os.path.join(os.path.dirname(__file__), cfg.text_input)
    cfg.text_input = prepare_input_plus(cfg.text_input, cfg.language)
    # api_key:
    api_key = cfg.api_key if "api_key" in cfg else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Need an API Key.")
    # in_context: instruction: / data_path:->examples
    examples = None
    if cfg.in_context:
        examples = prepare_examples(cfg.task, cfg.language, cfg.data_path)

    # <02: Build Prompt>
    # ie_prompter = PromptCraft(
    #     task=cfg.task,
    #     language=cfg.language,
    #     in_context=cfg.in_context,
    #     instruction=cfg.instruction if "instruction" in cfg else None,
    #     example=examples
    # )
    # prompt = ie_prompter.build_prompt(
    #     prompt=cfg.text_input,
    #     domain=cfg.domain if "domain" in cfg else None,
    #     labels=cfg.labels if "labels" in cfg else None,
    #     head_entity=cfg.head_entity if "head_entity" in cfg else None,
    #     head_type=cfg.head_type if "head_type" in cfg else None,
    #     tail_entity=cfg.tail_entity if "tail_entity" in cfg else None,
    #     tail_type=cfg.tail_type if "tail_type" in cfg else None,
    # )
    prompt = cfg.text_input
    # print("Your final customized-prompt: " + prompt)

    # return

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
    print(response)
    # print("Model Response: " + response)


if __name__ == "__main__":
    main()
