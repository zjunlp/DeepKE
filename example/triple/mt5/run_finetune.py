import logging
import os
import sys
import numpy as np
from datasets import Dataset

import random


import torch
import json
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from arguments import ModelArguments, DataTrainingArguments
from utils import get_extract_metrics_f1


os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger("__main__")



def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_name = training_args.output_dir.split('/')[-1]
    os.makedirs(training_args.logging_dir, exist_ok = True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(training_args.logging_dir, f'{log_name}.txt'), mode = 'w', encoding = 'utf-8')],
    )
    logger.setLevel(logging.INFO)

    
    logger.info("Detecting last checkpoint...")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.info(f"last_checkpoint: {last_checkpoint}")
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    logger.info("Set seed before initializing model....")
    seed_torch(training_args.seed)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()


    logger.info("Loading Dataset....")
    datasets = {}
    if data_args.train_file is not None:
        datasets["train"] = Dataset.from_json(data_args.train_file) 
    if data_args.validation_file is not None:
        datasets["validation"] = Dataset.from_json(data_args.validation_file)
    elif data_args.validation_file is None and data_args.train_file is not None:
        train_valid_datasets = datasets["train"].train_test_split(test_size=0.2, shuffle=True, seed=training_args.seed)
        datasets["train"] = train_valid_datasets["train"]
        datasets["validation"] = train_valid_datasets["test"]
    if data_args.test_file is not None:
        datasets["test"] = Dataset.from_json(data_args.test_file)


    logger.info("Loading Config....")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.max_length = data_args.max_target_length
    logger.info(f"Config: {config}")


    logger.info("Loading Tokenizer....")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    model = AutoModelForSeq2SeqLM.from_pretrained(           # define models
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )  
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        train_column_names = datasets["train"].column_names
    if training_args.do_eval:
        valid_column_names = datasets["validation"].column_names
    if training_args.do_predict:
        test_column_names = datasets["test"].column_names
    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.error(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    

    def preprocess_function_test(examples):
        inputs = [instruct+ inp  for inp, instruct in zip(examples["input"], examples["instruction"])]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        return model_inputs
    
    def preprocess_function(examples):
        targets = examples["output"]
        inputs = [instruct+ inp  for inp, instruct in zip(examples["input"], examples["instruction"])]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != tokenizer.pad_token_id else -100) for _label in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    


    logger.info("Start Data Preprocessing ...")
    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=valid_column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function_test,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=test_column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    logger.info("End Data Preprocessing ...")


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = get_extract_metrics_f1(golds_outtext=decoded_labels, preds_outtext=decoded_preds)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        max_length=data_args.max_source_length,
    )
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if model_args.from_checkpoint:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
        logger.info(f"checkpoint: {checkpoint}")

        logger.info("*** Training ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Save our tokenizer and create model card
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.save_model() 
        trainer.create_model_card()

        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        logger.info("***** Train results *****")
        logger.info(f"{metrics}")
        trainer.save_state()


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        logger.info("***** Evaluate results *****")
        logger.info(f"{metrics}")


    # Prediction
    if training_args.do_predict:
        logger.info("*** Prediction ***")
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        
        if training_args.predict_with_generate:
            preds = test_results.predictions
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            test_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            output_test_preds_file = os.path.join(training_args.output_dir, "test_preds.json")
            with open(output_test_preds_file, "w") as writer:
                for pred in test_preds:
                    writer.write(json.dumps({"output": pred}, ensure_ascii=False)+"\n")

        metrics = test_results.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["eval_samples"] = min(max_test_samples, len(test_dataset))

        logger.info("***** Prediction results *****")
        logger.info(f"{metrics}")



if __name__ == "__main__":
    main()

