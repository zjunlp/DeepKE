"""
run
"""


import os
import logging

import hydra
import openai

from utils.preprocess import prepare_examples

# 删除：
from easyinstruct.prompts import IEPrompt


logger = logging.getLogger(__name__)  # 日志记录


# Hydra 会自动解析目标 YAML 文件，将其中的每个字段变为 cfg 对象的属性。因此，在代码中你可以直接使用 cfg.api_key 或 cfg.task 等来获取配置文件中的值。
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # cwd:
    cfg.cwd = hydra.utils.get_original_cwd()  # 当前工作目录 Current Working Directory
    # api_key:
    if not cfg.api_key:
        raise ValueError("Need an API Key.")
    os.environ['OPENAI_API_KEY'] = cfg.api_key
    # engine：
    if cfg.engine not in ["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001", "gpt-3.5-turbo"]:
        raise ValueError("The OpenAI model is not supported now.")
    # zero_shot:
    examples = None
    if not cfg.zero_shot:
        examples = prepare_examples(cfg.data_path, cfg.task, cfg.language)
    # text_input:
    text = cfg.text_input

    # :
    ie_prompter = IEPrompt(cfg.task)
    if cfg.task == 're':
        ie_prompter.build_prompt(
            prompt=text,
            head_entity=cfg.head_entity,
            head_type=cfg.head_type,
            tail_entity=cfg.tail_entity,
            tail_type=cfg.tail_type,
            language=cfg.language,
            instruction=cfg.instruction,
            in_context=not cfg.zero_shot,
            domain=cfg.domain,
            labels=cfg.labels,
            examples=examples
        )
    else:
        ie_prompter.build_prompt(
            prompt=text,
            language=cfg.language,
            instruction=cfg.instruction,
            in_context=not cfg.zero_shot,
            domain=cfg.domain,
            labels=cfg.labels,
            examples=examples
        )

    result = ie_prompter.get_openai_result()
    logger.info(result)


if __name__ == '__main__':
    main()
