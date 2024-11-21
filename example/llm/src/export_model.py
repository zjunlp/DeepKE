import os
import argparse
from args.parser import get_infer_args
from model.loader import load_model_and_tokenizer
from datamodule.template import get_template_and_fix_tokenizer
from typing import Any, Dict, Optional
from utils.logging import get_logger
from utils.general_utils import get_model_tokenizer_trainer, get_model_name
from utils.load_cmd import load_config_from_yaml


logger = get_logger(__name__)


def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, data_args, training_args, finetuning_args, _, _ = get_infer_args(args)
    model_args.model_name = get_model_name(model_args.model_name)

    model_class, tokenizer_class, trainer_class = get_model_tokenizer_trainer(model_args.model_name)
    model, tokenizer = load_model_and_tokenizer(
        model_class,
        tokenizer_class,
        model_args,
        finetuning_args,
        training_args.do_train,
        stage="sft",
    )
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)
    print("tokenizer.eos_token", tokenizer.eos_token)

    model.config.use_cache = True
    model.save_pretrained(finetuning_args.export_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.padding_side = "left" # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(finetuning_args.export_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file")
    parser.add_argument('--model_name_or_path', type=str, default='models/llama-3-8b-Instruct', help="Path to model or model name")
    parser.add_argument('--checkpoint_dir', type=str, default='lora_results/llama3-v1/checkpoint-xxx', help="Checkpoint directory")
    parser.add_argument('--export_dir', type=str, default='lora_results/llama3-v1/llama3-v1', help="Directory to export the model")
    parser.add_argument('--stage', type=str, default='sft', help="Stage of the model")
    parser.add_argument('--model_name', type=str, default='llama', help="Model name")
    parser.add_argument('--template', type=str, default='llama3', help="Template name")
    parser.add_argument('--output_dir', type=str, default='lora_results/test', help="Output directory")
    args = parser.parse_args()

    if args.config:
        # if user specifies a config file, load the config from the file
        config = load_config_from_yaml(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


if __name__ == "__main__":
    args = set_args()
    export_model(args)
