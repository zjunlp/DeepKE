import os
import json
import argparse
import logging
from easyinstruct.prompts import IEPrompt
from .preprocess import prepare_examples

logger = logging.getLogger(__name__)

def add_args(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_input', 
                        type=str, 
                        default='My name is Ada.',
                        help='The input text to be processed.')
    parser.add_argument('--task', 
                        type=str, 
                        default='ner', 
                        choices=['ner', 're', 'ee', 'rte', 'da'], 
                        help='ner: Named Entity Recognition, re: Relation Extraction, ee: Event Extraction, rte: Recognizing Textual Entailment, da: Dialogue Act')
    parser.add_argument('--language', 
                        type=str, 
                        default='en', 
                        choices=['en', 'ch'],
                        help='en: English, ch: Chinese')
    parser.add_argument('--zero_shot', 
                        action='store_true', 
                        help='Whether to use zero-shot learning.')
    parser.add_argument('--head_entity', 
                        type=str, 
                        default='', 
                        help='The head entity of the relation.')
    parser.add_argument('--head_type', 
                        type=str, 
                        default='', 
                        help='The head entity type of the relation.')
    parser.add_argument('--tail_entity', 
                        type=str, 
                        default='',
                        help='The tail entity of the relation.')
    parser.add_argument('--tail_type', 
                        type=str, 
                        default='',
                        help='The tail entity type of the relation.')
    parser.add_argument('--data_path', 
                        type=str, 
                        default='./data',
                        help='The path of the data.')
    parser.add_argument('--api_key', 
                        type=str, 
                        default=None,
                        help='The API key of OpenAI.')
    parser.add_argument('--engine', 
                        type=str, 
                        default='text-davinci-003', 
                        choices=['text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'gpt-3.5-turbo'],
                        help='The engine of OpenAI.')
    parser.add_argument('--instruction', 
                        type=str, 
                        default=None,
                        help='The instruction of the task.')
    parser.add_argument('--domain', 
                        type=str, 
                        default=None,
                        help='The domain of the task.')
    parser.add_argument('--labels', 
                        type=str, 
                        default=None,
                        help='The labels of the task.')
    cfg = parser.parse_args()
    return cfg


def main(cfg):
    text = cfg.text_input

    if not cfg.api_key:
        raise ValueError("Need an API Key.")
    if cfg.engine not in ["text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001", "gpt-3.5-turbo"]:
        raise ValueError("The OpenAI model is not supported now.")

    os.environ['OPENAI_API_KEY'] = cfg.api_key

    ie_prompter = IEPrompt(cfg.task)

    examples = None
    if not cfg.zero_shot:
        examples = prepare_examples(cfg.data_path, cfg.task, cfg.language)

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
    cfg = add_args()
    main(cfg=cfg)