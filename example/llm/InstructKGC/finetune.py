import os
import random
import numpy as np
import sys
import argparse
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
    AutoTokenizer, 
    AutoModel, 
    Trainer,
    TrainingArguments, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq, 
)

from utils import MODEL_DICT
from utils.prompter import Prompter
from utils.llama import coll_fn_llama
from utils.chatglm import coll_fn_glm, ChatGLMDataCollatorForSeq2Seq

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


def get_format_time():
    current_time = int(time.time())
    localtime = time.localtime(current_time)
    dt = time.strftime('%Y:%m:%d %H:%M:%S', localtime)
    return dt


def get_model_name(model_name):
    model_name = model_name.lower()
    for key, values in MODEL_DICT.items():
        for v in values:
            if v in model_name:
                return key
    return ""


def add_args():
    parse = argparse.ArgumentParser()
    # model/data params
    parse.add_argument('--base_model', type=str, default=None)
    parse.add_argument('--train_path', type=str, default=None)
    parse.add_argument('--valid_path', type=str, default=None)
    parse.add_argument('--output_dir', type=str, default=None)
    parse.add_argument('--batch_size', type=int, default=80)

    # training hyperparams
    parse.add_argument('--micro_train_batch_size', type=int, default=10)
    parse.add_argument('--micro_eval_batch_size', type=int, default=10)
    parse.add_argument('--eval_save_steps', type=int, default=100)
    parse.add_argument('--save_total_limit', type=int, default=10)
    parse.add_argument('--num_epochs', type=int, default=3)
    parse.add_argument('--learning_rate', type=float, default=3e-4)
    parse.add_argument('--cutoff_len', type=int, default=256)
    parse.add_argument('--val_set_size', type=int, default=1000)
    parse.add_argument('--preprocessing_num_workers', type=int, default=8)
    parse.add_argument('--resume_from_checkpoint', type=str, default=None, help="either training checkpoint or final adapter")
    parse.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")

    # lora hyperparams
    parse.add_argument('--lora_r', type=int, default=8)
    parse.add_argument('--lora_alpha', type=int, default=16)
    parse.add_argument('--lora_dropout', type=float, default=0.05)
    parse.add_argument('--lora_target_modules', type=str, default="['q_proj','v_proj']")

    # llm hyperparams
    parse.add_argument('--train_on_inputs', action='store_true', help="if False, masks out inputs in loss")
    parse.add_argument('--add_eos_token', action='store_true')
    parse.add_argument('--group_by_length', action='store_true', help="faster, but produces an odd training loss curve")

    return parse


def train(options):
    seed_torch(42)
    options.lora_target_modules = eval(options.lora_target_modules)
    os.makedirs(options.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(options.output_dir, 'log.txt'), mode = 'w', encoding = 'utf-8')],
    )
    logger.setLevel(logging.INFO)
    start_time = get_format_time()
    logger.info(f"Start Time: {start_time}")


    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = options.batch_size // options.micro_train_batch_size
    ddp = world_size != 1
    assert (options.base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Options: {options}")
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    logger.info(f"world_size:{world_size}\nddp: {ddp}\ngradient_accumulation_steps: {gradient_accumulation_steps}\ndevice_map: {device_map}\n")

    
    prompter = Prompter(options.prompt_template_name)
    config = LoraConfig(
        r=options.lora_r,
        lora_alpha=options.lora_alpha,
        target_modules=options.lora_target_modules,
        lora_dropout=options.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"config: {config}")
    tokenizer = AutoTokenizer.from_pretrained(options.base_model, trust_remote_code=True)    
    model = AutoModel.from_pretrained(
        options.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = prepare_model_for_int8_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    if options.train_path is not None:
        train_data = Dataset.from_json(options.train_path)
    if options.valid_path is not None:
        valid_data = Dataset.from_json(options.valid_path)
    elif options.valid_path is None and options.train_path is not None:
        train_val = train_data.train_test_split(test_size=options.val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"]
        valid_data = train_val["test"]
    
    
    model_name = get_model_name(options.base_model)
    if model_name not in MODEL_DICT:
        raise KeyError
    coll_fn = coll_fn_llama
    fn_kwargs = {"prompter":prompter, "tokenizer":tokenizer, "options":options}
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    ModifyTrainer = Trainer
    ModifyArgument = TrainingArguments
    if model_name == 'falcon':
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif model_name == 'llama':
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    elif model_name == "chatglm":
        coll_fn = coll_fn_glm
        ModifyTrainer = Seq2SeqTrainer
        ModifyArgument = Seq2SeqTrainingArguments
        data_collator = ChatGLMDataCollatorForSeq2Seq(tokenizer=tokenizer)
    logger.info(f"model_name: {model_name}\ncoll_fn: {coll_fn}\ndata_collator: {data_collator}\nModifyTrainer: {ModifyTrainer}\n")
    logger.info(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")



    train_data = train_data.shuffle().map(
        coll_fn_glm, 
        num_proc=options.preprocessing_num_workers, 
        remove_columns=train_data.column_names,
        load_from_cache_file=True,
        fn_kwargs=fn_kwargs,
    )
    valid_data = valid_data.shuffle().map(
        coll_fn_glm, 
        num_proc=options.preprocessing_num_workers, 
        remove_columns=valid_data.column_names,
        load_from_cache_file=True,
        fn_kwargs=fn_kwargs,
    )


    resume_from_checkpoint = options.resume_from_checkpoint
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


    model.config.use_cache = False
    trainer = ModifyTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=ModifyArgument(
            per_device_train_batch_size=options.micro_train_batch_size,
            per_device_eval_batch_size=options.micro_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0.06,
            num_train_epochs=options.num_epochs,
            learning_rate=options.learning_rate,
            fp16=True,
            logging_steps=50,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=options.output_dir,
            save_total_limit=options.save_total_limit,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=options.group_by_length,
            remove_unused_columns=False,
        ),
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
    model.save_pretrained(options.output_dir)
    trainer.save_state()
    end_time = get_format_time()
    logger.info(f"End Time: {end_time}")




def main():
    parse = add_args()
    options = parse.parse_args()
    train(options)




if __name__ == "__main__":
    main()
