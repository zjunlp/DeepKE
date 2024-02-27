#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training dataset."})
    valid_file: str = field(default=None, metadata={"help": "Path to the validation dataset."})
    predict_file: str = field(default=None, metadata={"help": "Path to the validation dataset."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."},)
    overwrite_cache: Optional[bool] = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets."})
    cache_path: Optional[str] = field(default=None, metadata={"help": "Path to save or load the preprocessed datasets."})

    template: str = field(default="alpaca", metadata={"help": "The prompt template to use, will default to alpaca."})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt to add before the user query. Use `|` to separate multiple prompts in training."})
    max_source_length: Optional[int] = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    cutoff_len: Optional[int] = field(default=1024, metadata={"help": "The maximum length of the model inputs after tokenization."})
    val_set_size: Optional[int] = field(default=1000, metadata={"help": "The maximum prefix length."})

    pad_to_max_length: bool = field(default=False, metadata={"help": "Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More efficient on GPU but very bad for TPU."})
    ignore_pad_token_for_loss: bool = field(default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."})
    train_on_prompt: bool = field(default=False, metadata={"help": "If False, masks out inputs in loss."})
    language: Optional[str] = field(default="zh", metadata={"help": "The language."})
    id_text: str = field(default='input')

