import os
import sys
import json
import fire
import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
)
from peft.utils import WEIGHTS_NAME
from transformers import GenerationConfig, AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM
from utils.prompter import Prompter
from utils import MODEL_DICT

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["WANDB_DISABLED"] = "true"


def get_model_name(model_name):
    model_name = model_name.lower()
    for key, values in MODEL_DICT.items():
        for v in values:
            if v in model_name:
                return key
    return ""


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "",
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
    input_file: str = "data/valid.json",
    output_file: str = "result/valid_alpaca_result.json",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    model_name = get_model_name(base_model)


    prompter = Prompter(prompt_template)
    if model_name == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )


    if lora_weights is not None:
        if load_8bit:
            config = LoraConfig.from_pretrained(lora_weights)
            model = get_peft_model(model, config) 
            adapters_weights = torch.load(os.path.join(lora_weights, WEIGHTS_NAME), map_location=model.device)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
            )

    print("model_name", model_name)
    if model_name == 'llama':
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    print(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")


    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        num_beams=1,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )

        generation_output = generation_output.sequences[0]
        output = tokenizer.decode(generation_output, skip_special_tokens=True)
        output = prompter.get_response(output)
        return output

    records = []
    with open(input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            records.append(data)

    with open(output_file, "w") as writer:
        for i, record in enumerate(records):
            result = evaluate(instruction=record['instruction'], input=record['input'])
            print(i, result)
            record['output'] = result
            writer.write(json.dumps(record, ensure_ascii=False)+'\n')  


if __name__ == "__main__":
    fire.Fire(main)
