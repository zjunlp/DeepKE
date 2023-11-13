#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    full_finetune: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
    cache_dir: Optional[str] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "The path to a folder with a valid checkpoint for your model."},)
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={"help": "Whether or not to load the best model found during training at the end of training."},)
    
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})
    
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help":"Lora dropout."})
    lora_target_modules: Optional[str] = field(default=None, metadata={ "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."})
    loss_scale: float = field(default=1.0, metadata={"help": 'Loss scaling, positive power of 2 values can improve fp16 convergence.'})

    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    report_to: str = field(default='none', metadata={"help": "To use wandb or something else for reporting."})
    logging_steps: float = field(default=10, metadata={"help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    model_name: str = field(default="llama", metadata={"help": "Model name."})    # choices=["llama", "falcon", "baichuan", "chatglm", "moss", "alpaca", "vicuna", "zhixi"]
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Enables using Huggingface auth token from Git Credentials."})
    def __post_init__(self):
        assert self.model_name in ["llama", "falcon", "baichuan", "chatglm", "moss", "alpaca", "vicuna", "zhixi"],\
            f"""`{self.model_name}` is invalid. We only support model in ["llama", "falcon", "baichuan", "chatglm", "moss", "alpaca", "vicuna", "zhixi"]"""
      


@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training dataset."})
    valid_file: str = field(default=None, metadata={"help": "Path to the validation dataset."})
    max_source_length: Optional[int] = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    cutoff_len: Optional[int] = field(default=512, metadata={"help": "cutoff length"})
    val_set_size: Optional[int] = field(default=1000, metadata={"help": "The maximum prefix length."})

    prompt_template_name: str = field(default="alpaca", metadata={"help": "The prompt template to use, will default to alpaca."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."},)
    pad_to_max_length: bool = field(default=False, metadata={"help": "Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More efficient on GPU but very bad for TPU."})
    ignore_pad_token_for_loss: bool = field(default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."})
    train_on_inputs: bool = field(default=False, metadata={"help": "If False, masks out inputs in loss."})


    
@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops if predict_with_generate is set."})
    min_new_tokens : Optional[int] = field(default=None, metadata={"help": "Minimum number of new tokens to generate."})

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)
