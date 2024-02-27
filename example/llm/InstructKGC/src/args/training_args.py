#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "The path to a folder with a valid checkpoint for your model."},)
    load_best_model_at_end: Optional[bool] = field(default=False, metadata={"help": "Whether or not to load the best model found during training at the end of training."},)
    loss_scale: float = field(default=1.0, metadata={"help": 'Loss scaling, positive power of 2 values can improve fp16 convergence.'})

    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    report_to: str = field(default='none', metadata={"help": "To use wandb or something else for reporting."})
    logging_steps: float = field(default=2, metadata={"help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."})
