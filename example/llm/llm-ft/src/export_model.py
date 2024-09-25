from args.parser import get_infer_args
from model.loader import load_model_and_tokenizer
from datamodule.template import get_template_and_fix_tokenizer
from typing import Any, Dict, Optional
from utils.logging import get_logger
from utils.general_utils import get_model_tokenizer_trainer, get_model_name

logger = get_logger(__name__)


'''
python src/export_model.py \
    --model_name_or_path 'models/llama-3-8b-Instruct' \
    --checkpoint_dir 'lora_results/llama3-v1/checkpoint-xxx' \
    --export_dir 'lora_results/llama3-v1/llama3-v1' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama3' \
    --output_dir 'lora_results/test'

python src/export_model.py \
    --model_name_or_path 'models/Qwen__Qwen2-1.5B-Instruct' \
    --checkpoint_dir 'lora_results/qwen2-1.5b-v1/checkpoint-xxx' \
    --export_dir 'lora_results/qwen2-1.5b-v1/qwen2-1.5b-v1' \
    --stage 'sft' \
    --model_name 'qwe2' \
    --template 'qwen' \
    --output_dir 'lora_results/test'

python src/export_model.py \
    --model_name_or_path 'models/Baichuan2-13B-Chat' \
    --checkpoint_dir 'lora_results/baichuan2-13b-v1/checkpoint-xxx' \
    --export_dir 'lora_results/baichuan2-13b-v1/baichuan2-13b-v1' \
    --stage 'sft' \
    --model_name 'baichuan' \
    --template 'baichuan2' \
    --output_dir 'lora_results/test'
'''


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
        
if __name__ == "__main__":
    export_model()


