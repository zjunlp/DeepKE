import os
import sys
import logging
import argparse

from datasets import Dataset
import torch
from transformers import (
    HfArgumentParser,
    DataCollatorForSeq2Seq, 
    GenerationConfig,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from peft.tuners.lora import LoraLayer

from utils.general_utils import (
    LORA_TARGET_MODULES_DICT, 
    seed_torch, 
    get_model_tokenizer, 
    get_model_name, 
    get_format_time,
)
from utils.prompter import Prompter
from utils.args import DataArguments, ModelArguments, TrainingArguments, GenerationArguments
from datamodule import COLL_FN_DICT, DataCollatorForCPMBEE


os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger("__main__")



def get_specific(model_name, tokenizer, model, args):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )
    prompter = Prompter(args.prompt_template_name)
    coll_fn = COLL_FN_DICT[model_name]()
    if model_name == 'llama':
        model.config.pad_token_id = tokenizer.pad_token_id = 0 
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    elif model_name == "moss":
        model.config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == "cpm-bee":
        data_collator = DataCollatorForCPMBEE(tokenizer, args.cutoff_len)
    return coll_fn, data_collator, prompter



def get_accelerate_model(args, model_class):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    logger.info(f"compute_dtype:{compute_dtype}")
    model = model_class.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder="./cache",      # If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        quantization_config=BitsAndBytesConfig(     
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=compute_dtype

    if not args.full_finetune:    
        if args.bits in [4, 8]:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        print(f'adding LoRA modules...')
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        if args.resume_from_checkpoint is not None: 
            print("Loading adapters from checkpoint.")
            checkpoint_name = os.path.join(args.resume_from_checkpoint, "pytorch_model.bin")  
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin") 
                args.resume_from_checkpoint = (False  )
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
    '''
    for name, module in model.named_modules():
        #if isinstance(module, LoraLayer):
            #if args.bf16:
                #module = module.to(torch.bfloat16)
        #if 'norm' in name:
            #module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    '''
    return model, args



def verify_datatype(model):
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)



def train(model_args, data_args, training_args, args):
    logger.info(f"Start Time: {get_format_time()}")
    logger.info(f"model_args:{model_args}\ndata_args:{data_args}\ntraining_args:{training_args}\n")
    model_class, tokenizer_class, trainer_class = get_model_tokenizer(model_args.model_name)
    logger.info(f"model_class:{model_class}\ntokenizer_class:{tokenizer_class}\ntrainer_class:{trainer_class}\n")
    tokenizer = tokenizer_class.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=args.trust_remote_code
    )  
    model, args = get_accelerate_model(args, model_class)
    model.config.use_cache = False
    model.print_trainable_parameters()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        training_args.ddp_find_unused_parameters = False
    else:
        training_args.ddp_find_unused_parameters = None


    if data_args.train_file is not None:
        train_data = Dataset.from_json(data_args.train_file)
    if data_args.valid_file is not None:
        valid_data = Dataset.from_json(data_args.valid_file)
    else:
        if training_args.do_eval and data_args.val_set_size > 0:
            train_val = train_data.train_test_split(test_size=data_args.val_set_size, shuffle=True, seed=42)
            train_data = train_val["train"]
            valid_data = train_val["test"]
    
    
    coll_fn, data_collator, prompter = get_specific(model_args.model_name, tokenizer, model, data_args)
    fn_kwargs = {"prompter":prompter, "tokenizer":tokenizer, "options":data_args}
    logger.info(f"coll_fn:{coll_fn}\ndata_collator:{data_collator}\nprompter:{prompter}\n")
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


    trainer = trainer_class(
        model=model,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        args=training_args,
        data_collator=data_collator,
    )


    all_metrics = {}
    verify_datatype(model)
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        print("args.resume_from_checkpoint", args.resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    logger.info(f"End Time: {get_format_time()}")



def main():
    hfparser = HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args =  hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))


    model_args.model_name = get_model_name(model_args.model_name)
    if training_args.lora_target_modules:
        training_args.lora_target_modules = eval(training_args.lora_target_modules)
    else:
        training_args.lora_target_modules = LORA_TARGET_MODULES_DICT[model_args.model_name]
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )


    seed_torch(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(training_args.output_dir, 'log.txt'), mode = 'w', encoding = 'utf-8')],
    )
    logger.setLevel(logging.INFO)


    train(model_args, data_args, training_args, args)



if __name__ == "__main__":
    main()
