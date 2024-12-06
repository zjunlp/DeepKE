import argparse
import yaml
import subprocess
import os

def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_command_from_yaml(config, base_command):
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                base_command.append(f'--{key}')
        else:
            base_command.append(f'--{key}={value}')
    return base_command

def main():
    # parse the mood argument
    parser = argparse.ArgumentParser(description="choose the training mood")
    parser.add_argument(
        '--mood',
        type=str,
        choices=['lora', 'pt', 'infer', 'infer_vllm', 'icl', 'icl_batch'],
        required=True,
        help="choose the training mood, please read README.md for more details"
    )
    args = parser.parse_args()

    # according to the mood, choose the corresponding yaml config file and base command
    if args.mood == 'lora':
        yaml_config_path = 'examples/infer/fine_llama.yaml'
        base_command = ["python", "src/finetune.py"]

    elif args.mood == 'pt':
        yaml_config_path = 'examples/infer/fine_chatglm_pt.yaml'
        base_command = ["deepspeed", "--include", "localhost:0", "src/finetune_chatglm_pt.py"]

    elif args.mood == 'infer':
        yaml_config_path = 'examples/infer/infer_llama.yaml'
        base_command = ["python", "src/inference.py"]

    elif args.mood == 'infer_vllm':
        yaml_config_path = 'examples/infer/infer_baichuan_vllm.yaml'
        base_command = ["python", "src/inference_vllm.py"]

    elif args.mood == 'icl':
        yaml_config_path = 'examples/incontext_learning/da_qwen.yaml'
        base_command = ["python", "src/incontext_learning.py"]

    elif args.mood == 'icl_batch':
        yaml_config_path = 'examples/incontext_learning/re_chatgpt_plus.yaml'
        base_command = ["python", "src/incontext_learning_plus.py"]

    config = load_yaml_config(yaml_config_path)
    command = generate_command_from_yaml(config, base_command)
    subprocess.run(command)


if __name__ == "__main__":
    main()
