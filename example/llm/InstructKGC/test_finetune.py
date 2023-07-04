import os
import random
import numpy as np
import sys
import logging
import time
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from datasets import Dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer, 
    AutoModel, 
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq, 
)

from utils import MODEL_DICT
from utils.prompter import Prompter
from utils.args import DataTrainingArguments, ModelArguments
from utils.llama import coll_fn_llama
from utils.chatglm import coll_fn_chatglm

os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger("__main__")


def get_model_name(model_name):
    model_name = model_name.lower()
    for key, values in MODEL_DICT.items():
        for v in values:
            if v == model_name:
                return key
    return ""



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


def get_format_time():
    current_time = int(time.time())
    localtime = time.localtime(current_time)
    dt = time.strftime('%Y:%m:%d %H:%M:%S', localtime)
    return dt


def get_specific(prompt_template, model_name, tokenizer, model):
    if model_name == 'llama':
        MyTrainer = Trainer
        model.config.pad_token_id = tokenizer.pad_token_id = 0 
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
        coll_fn = coll_fn_llama()
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True
        )
        prompter = Prompter(prompt_template)
    elif model_name == "chatglm":
        MyTrainer = Seq2SeqTrainer
        coll_fn = coll_fn_chatglm()
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True
        )
        prompter = Prompter(prompt_template)
    else:
        raise KeyError
    return MyTrainer, coll_fn, data_collator, prompter


def get_model_tokenizer(model_name):
    if model_name == 'llama':
        return LlamaForCausalLM, LlamaTokenizer
    elif model_name == "chatglm":
        return AutoModel, AutoTokenizer
    else:
        raise KeyError


def train(model_args, data_args, training_args):
    seed_torch(42)
    os.makedirs(training_args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(training_args.output_dir, 'log.txt'), mode = 'w', encoding = 'utf-8')],
    )
    logger.setLevel(logging.INFO)
    model_name = get_model_name(model_args.model_name)
    if model_name not in MODEL_DICT:
        raise KeyError
    logger.info(f"model_name: {model_name}")
    start_time = get_format_time()
    logger.info(f"Start Time: {start_time}")


    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    ddp = world_size != 1
    assert (model_args.base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Model args: {model_args}\nData args: {data_args}\nTraining args: {training_args}")
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    logger.info(f"world_size:{world_size}\nddp: {ddp}\ngradient_accumulation_steps: {training_args.gradient_accumulation_steps}\ndevice_map: {device_map}\n")

    
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"config: {config}")


    MyModel, MyTokenizer = get_model_tokenizer(model_name)
    tokenizer = MyTokenizer.from_pretrained(
        model_args.base_model, 
        trust_remote_code=True)    
    model = MyModel.from_pretrained(
        model_args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = prepare_model_for_int8_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    if data_args.train_path is not None:
        train_data = Dataset.from_json(data_args.train_path)
    if data_args.valid_path is not None:
        valid_data = Dataset.from_json(data_args.valid_path)
    else:
        if training_args.do_eval and data_args.val_set_size > 0:
            train_val = train_data.train_test_split(test_size=data_args.val_set_size, shuffle=True, seed=42)
            train_data = train_val["train"]
            valid_data = train_val["test"]
    
    
    MyTrainer, coll_fn, data_collator, prompter = get_specific(data_args.prompt_template_name, model_args.model_name, tokenizer, model)
    fn_kwargs = {"prompter":prompter, "tokenizer":tokenizer, "options":data_args}
    logger.info(f"coll_fn: {coll_fn}\ndata_collator: {data_collator}\nMyTrainer: {MyTrainer}\n")
    logger.info(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")


    if training_args.do_train:
        train_data = train_data.shuffle().map(
            coll_fn, 
            num_proc=data_args.preprocessing_num_workers, 
            remove_columns=train_data.column_names,
            load_from_cache_file=True,
            fn_kwargs=fn_kwargs,
        )
    if training_args.do_eval:
        valid_data = valid_data.shuffle().map(
            coll_fn, 
            num_proc=data_args.preprocessing_num_workers, 
            remove_columns=valid_data.column_names,
            load_from_cache_file=True,
            fn_kwargs=fn_kwargs,
        )


    resume_from_checkpoint = training_args.resume_from_checkpoint
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
        logger.info(f"model.is_parallelizable: {model.is_parallelizable}\nmodel.model_parallel: {model.model_parallel}")
    if ddp:   
        training_args.ddp_find_unused_parameters = False
    else:
        training_args.ddp_find_unused_parameters = None


    model.config.use_cache = False
    trainer = MyTrainer(
        model=model,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        args=training_args,
        data_collator=data_collator,
    )
    

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    logger.info("*** Training ***")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = train_result.metrics
    logger.info("***** Train results *****")
    logger.info(f"{metrics}")
    model.save_pretrained(training_args.output_dir)
    trainer.save_state()
    end_time = get_format_time()
    logger.info(f"End Time: {end_time}")




def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.lora_target_modules = eval(model_args.lora_target_modules)

    train(model_args, data_args, training_args)




if __name__ == "__main__":
    main()
