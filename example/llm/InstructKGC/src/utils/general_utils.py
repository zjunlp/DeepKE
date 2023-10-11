import os
import random
import time
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForCausalLM, 
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    Seq2SeqTrainer, 
)

MODEL_DICT = {
    "llama":["llama", "alpaca", "vicuna", "zhixi"],
    "falcon":["falcon"],
    "baichuan":["baichuan"],
    "chatglm":["chatglm"],
    "moss": ["moss"],
}

LORA_TARGET_MODULES_DICT = {
    "llama":['q_proj','v_proj'],
    "falcon":["query_key_value"],
    "baichuan":['W_pack','o_proj','gate_proj','down_proj','up_proj'],
    "chatglm":["query_key_value"],
    "moss": ['q_proj', 'v_proj'],
}

def get_model_tokenizer(model_name):
    if model_name == 'llama':
        return LlamaForCausalLM, LlamaTokenizer, Trainer
    elif model_name == 'chatglm':
        return AutoModel, AutoTokenizer, Seq2SeqTrainer
    else:
        return AutoModelForCausalLM, AutoTokenizer, Trainer
    

def get_model_name(model_name):
    model_name = model_name.lower()
    for key, values in MODEL_DICT.items():
        for v in values:
            if v == model_name:
                return key
    return "other"


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



