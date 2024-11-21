#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class GenerationArguments:
    max_length: Optional[int] = field(default=512, metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."})
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

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args


@dataclass
class InferArguments:
    mode: str = field(default='w')
    gen_mode: str = field(default='greedy', metadata={"help": "gen_mode."})
    swap_space: int = field(default=4, metadata={"help": "CPU swap space size (GiB) per GPU"})
    gpu_memory_utilization: float = field(default=0.90, metadata={"help": "the percentage of GPU memory to be used for the model executor"})
    
    input_file: str = field(default=None, metadata={"help": "Path to the input file."})
    output_file: str = field(default=None, metadata={"help": "Path to the output file."})
    batch_size: int = field(default=16, metadata={"help": "batch size"})


