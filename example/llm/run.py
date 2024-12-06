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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mood',
        type=str,
        required=True,
        help="choose the training mood, please read README.md for more details"
    )
    args = parser.parse_args()

    # according to the mood, choose the corresponding yaml config file and base command
    if args.mood == 'test':
        print("this is test\n")
        return

    elif args.mood == 'sft_lora':
        yaml_config_path = 'examples/infer/fine_llama.yaml'
        base_command = ["python", "src/finetune.py"]

    elif args.mood == 'sft_pt':
        yaml_config_path = ''
        base_command = []

    elif args.mood == 'infer':
        yaml_config_path = 'examples/infer/infer_llama.yaml'
        base_command = ["python", "src/inference.py"]

    elif args.mood == 'icl':
        yaml_config_path = 'examples/icl/icl_qwen.yaml'
        base_command = ["python", "src/iclearning.py"]

    elif args.mood == 'icl_batch':
        yaml_config_path = 'examples/icl/icl_batch_chatgpt.yaml'
        base_command = ["python", "src/iclearning_batch.py"]

    config = load_yaml_config(yaml_config_path)
    command = generate_command_from_yaml(config, base_command)
    subprocess.run(command)


if __name__ == "__main__":
    main()
