#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    base_model: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    model_name: str = field(
        default="llama", metadata={"help": "Model name."}
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    lora_target_modules: Optional[str] = field(
        default="['q_proj','v_proj']",
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )




@dataclass
class DataTrainingArguments:
    train_path: str = field(
        default=None, metadata={"help": "Path to the training dataset."}
    )
    valid_path: str = field(
        default=None, metadata={"help": "Path to the validation dataset."}
    )
    prompt_template_name: str = field(
        default="alpaca", metadata={"help": "The prompt template to use, will default to alpaca."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_set_size: Optional[int] = field(
        default=1000,
        metadata={
            "help": "The maximum prefix length."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    train_on_inputs: bool = field(
        default=False,
        metadata={
            "help": "If False, masks out inputs in loss."
        },
    )
    add_eos_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    
