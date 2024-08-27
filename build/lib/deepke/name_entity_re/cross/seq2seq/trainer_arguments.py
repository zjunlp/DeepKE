#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    from_checkpoint: bool = field(
        default=False, metadata={"help": "Whether load from checkpoint to continue learning"}
    )
    model_ckpt_path: str = field(
            default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
                    "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default='text',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    record_column: Optional[str] = field(
        default='record',
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocess: bool = field(
        default=True,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessed_folder: Optional[str] = field(
        default=None,
        metadata={
            "help": "Folder to preprocessed data"
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_prefix_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum prefix length."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
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
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    meta_negative: int = field(
        default=-1, metadata={"help": "Negative Schema Number in Training."}
    )
    ordered_prompt: bool = field(
        default=True,
        metadata={
            "help": "Whether to sort the spot prompt and asoc prompt or not."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

    decoding_format: str = field(
        default='spotasoc',
        metadata={"help": "Decoding Format"}
    )
    record_schema: str = field(
        default=None, metadata={"help": "The input event schema file."}
    )
    spot_noise: float = field(
        default=0., metadata={"help": "The noise rate of null spot."}
    )
    asoc_noise: float = field(
        default=0., metadata={"help": "The noise rate of null asoc."}
    )
    meta_positive_rate: float = field(
        default=1., metadata={"help": "The keep rate of positive spot."}
    )
    eval_subset: str = field(default='validation')
    

@dataclass
class PromptArguments:
    use_prefix: bool = field(
        default=False,
        metadata={"help": "use prompt."},
    )
    freeze_plm: bool = field(
        default=False,
        metadata={"help": "freeze plm."},
    )
    prompt_len: int = field(
        default=80,
        metadata={"help": "prompt len."},
    )
    prompt_dim: int = field(
        default=800,
        metadata={"help": "prompt dim."},
    )
    source_prefix_path: str = field(
        default=None,
        metadata={"help": "source prefix path"},
    )
    target_prefix_path: str = field(
        default=None,
        metadata={"help": "target prefix path"},
    )
    prefix_fusion_way: str = field(
        default='add',
        metadata={"help": "[add, concat, concat_lstm]"},
    )
    freeze_prefix: bool = field(
        default=False,
        metadata={"help": "freeze prefix."},
    )
    use_adapter: bool = field(
        default=False,
        metadata={"help": "whether use adapter."},
    )
    save_prefix: bool = field(
        default=False,
        metadata={"help": "whether save prefix."},
    )
    save_label_word: bool = field(
        default=False,
        metadata={"help": "whether save label word."},
    )
    multi_source_path: str = field(
        default=None,
        metadata={"help": "the path of multi source."},
    )

