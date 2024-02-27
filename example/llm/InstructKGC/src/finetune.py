import os
import sys
sys.path.append('./')
import numpy as np


import torch
from transformers import DataCollatorForSeq2Seq


from utils.general_utils import (
    LORA_TARGET_MODULES_DICT, 
    seed_torch, 
    get_model_tokenizer_trainer, 
    get_model_name, 
    get_format_time,
)
from utils.logging import get_logger
from utils.constants import IGNORE_INDEX
from args.parser import get_train_args
from model.loader import load_model_and_tokenizer
from datamodule.get_datasets import load_train_datasets, process_datasets

os.environ["WANDB_DISABLED"] = "true"

logger = get_logger(__name__)


def train(model_args, data_args, training_args, finetuning_args, generating_args):
    logger.info(f"Start Time: {get_format_time()}")
    logger.info(f"model_args:{model_args}\ndata_args:{data_args}\ntraining_args:{training_args}\nfinetuning_args:{finetuning_args}\ngenerating_args:{generating_args}")
    # 获得特定于model name的特有类
    model_class, tokenizer_class, trainer_class = get_model_tokenizer_trainer(model_args.model_name)
    logger.info(f"model_class:{model_class}\ntokenizer_class:{tokenizer_class}\ntrainer_class:{trainer_class}\n")
    

    model, tokenizer = load_model_and_tokenizer(
        model_class,
        tokenizer_class,
        model_args,
        finetuning_args,
        training_args.do_train,
        stage="sft",
    )     # 获得处理包装后的模型
    logger.info(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")


    train_data_dict, train_data, valid_data = load_train_datasets(training_args, data_args)
    train_data, valid_data = process_datasets(
        training_args, 
        data_args, 
        finetuning_args, 
        tokenizer, 
        train_data_dict, 
        train_data, 
        valid_data,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4 if tokenizer.padding_side == "left" else None, # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )


    trainer = trainer_class(
        model=model,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        args=training_args,
        data_collator=data_collator,
    )

    all_metrics = {}
    if training_args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        logger.info(f"resume_from_checkpoint: {training_args.resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    logger.info(f"End Time: {get_format_time()}")



def main(args=None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # model_name映射
    model_args.model_name = get_model_name(model_args.model_name)
    
    # 如果为None则通过model_name加载默认的lora_target_modules, 否则加载传入的 
    if finetuning_args.lora_target_modules:  
        finetuning_args.lora_target_modules = eval(finetuning_args.lora_target_modules)
    else:
        finetuning_args.lora_target_modules = LORA_TARGET_MODULES_DICT[model_args.model_name]

    seed_torch(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)
    train(model_args, data_args, training_args, finetuning_args, generating_args)



if __name__ == "__main__":
    main()
