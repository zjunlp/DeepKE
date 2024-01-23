import os
import argparse

import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    LlamaForCausalLM.save_pretrained(
        base_model, args.output_dir, state_dict=deloreanized_sd, max_shard_size="1GB"
    )

    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':

    main()