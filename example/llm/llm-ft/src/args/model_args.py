#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    model_name: str = field(default="llama", metadata={"help": "Model name."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."})
    use_fast_tokenizer: Optional[bool] = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Enables using Huggingface auth token from Git Credentials."})
    model_revision: Optional[str] = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    split_special_tokens: Optional[bool] = field(default=False, metadata={"help": "Whether or not the special tokens should be split during the tokenization process."})
    
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})

    checkpoint_dir: Optional[str] = field(default=None, metadata={"help": "Path to the directory(s) containing the model checkpoints as well as the configurations."})

    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.checkpoint_dir is not None: # support merging multiple lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
