"""
interact.py

Mainly provide a user interaction interface.

Now:
has been discarded, will be re-implemented in the future.
"""


import re


MODEL_LIST = [
    "LLaMA",
    "Qwen",
    "MiniCPM",
    "ChatGLM",
    "ChatGPT",
    "DeepSeek",
]

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


class FaceDesign:
    """
    A class to provide a user interaction interface for configuring and executing NLP tasks.

    This class facilitates user interaction through command-line input,
    allowing for easy configuration of settings required for various NLP tasks.
    It includes methods to validate configurations and ensure they meet the necessary criteria.
    """
    def __init__(self):
        self.config = {}
        method = input("Please choose (1: cmd input, 2: visualization): ").strip()
        if method == '1':
            self.cmd_cfg()
        elif method == '2':
            print("Not yet...")
        else:
            raise ValueError("Please press '1' or '2' by your keyboard.")

    def cmd_cfg(self):
        """
        Prompts the user for configuration inputs via the command line and validates them.

        This method gathers required parameters for the NLP task from user input,
        ensuring that all necessary settings are provided and valid before execution.
        """
        # Primary Params of model
        while True:
            _ = input("Your Model: ")
            if _ in MODEL_LIST:
                self.config['engine'] = _
                break
            print("Your model is not supported now, we only support " + ", ".join(MODEL_LIST) + ". Try again please.")

        if self.config['engine'] in ["ChatGPT", "DeepSeek"]:
            self.config['api_key'] = input("Your API key: ")
            self.config['base_url'] = input("Your base url (can be skipped): ") or None

        self.config['model_id'] = input("Your open source name or path / Your closed model ID: ")

        # Advanced Params of model
        _ = input("Do you want use hidden model parameter? (y/n): ").strip().lower()
        if _ == 'y':
            self.config['temperature'] = float(input("Your temperature (can be skipped): ") or 0.7)
            self.config['top_p'] = float(input("Your Top-P (can be skipped): ") or 0.9)
            self.config['max_tokens'] = int(input("Your max tokens (can be skipped): ") or 512)
            self.config['stop'] = input("Your stop label: ") or None

        # Primary Params of task
        while True:
            _ = input("Your task: ")
            if _ in TASK_LIST:
                self.config['task'] = _
                break
            print("The task name should be one of ner, re, ee, rte for Information Extraction or da for Data Augmentation. Try again please.")

        while True:
            _ = input("Your language: ")
            if _ in LANGUAGE_LIST:
                self.config['language'] = _
                break
            print("Now we only support language of 'en' (English) and 'ch' (Chinese). Try again please.")

        while True:
            _ = input("Your text input: ")
            if _:
                self.config['text_input'] = _
                break
            print("Text Input can't be null")

        _ = input("Do you want use in-context learning? (y/n): ").strip().lower()
        if _ == 'y':
            self.config['in_context'] = True
            self.config['data_path'] = input("Your data path: ")
        else:
            self.config['in_context'] = False


        _ = input("(Not recommend) Do you want set instruction by yourself? (y/n): ").strip().lower()
        if _ == 'y':
            self.config['instruction'] = input("Your instruction: ")

        # Advanced Params of task
        if self.config['task'] in ["ner", "re", "ee", "rte"]:
            self.config['domain'] = input("Your domain (can be skipped): ") or None

        if self.config['task'] in ["ner", "re", "da"]:
            labels_input = input("Your labels (using ',' to split, can be skipped): ")
            self.config['labels'] = [label.strip() for label in re.split(r'[，,]', labels_input)] if labels_input else []

        if self.config['task'] == "re":
            self.config['head_entity'] = input("Your head entity: ") or None
            self.config['head_type'] = input("Your head type: ") or None
            self.config['tail_entity'] = input("Your tail entity: ") or None
            self.config['tail_type'] = input("Your tail type: ") or None


    def vis_cfg(self):
        """
        visual interface
        """
        pass


if __name__ == "__init__":
    pass
