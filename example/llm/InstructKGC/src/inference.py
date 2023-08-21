import os
import sys
import json
import argparse
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser, GenerationConfig, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
)
from peft.utils import WEIGHTS_NAME

from utils.prompter import Prompter
from utils.general_utils import get_model_tokenizer, get_model_name


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except: 
    pass

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class InferArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Model name or path."})
    model_name: str = field(default='llama', metadata={"help": "all supported models can be found in ./src/utils/general_utils.py:MODEL_DICT."})
    lora_weights: str = field(default=None, metadata={"help": "Path to the lora weights directory."})
    input_file: str = field(default=None, metadata={"help": "Path to the input file."})
    output_file: str = field(default=None, metadata={"help": "Path to the output file."})
    prompt_template_name: str = field(default='alpaca', metadata={"help": "Prompt template name. All prompt template can be found in ./templates"})
    
    max_source_length: int = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    max_target_length: int = field(default=256, metadata={"help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    cutoff_len: int = field(default=512, metadata={"help": "cutoff length"})
    
    trust_remote_code: bool = field(default=True, metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})
    train_on_inputs: bool = field(default=False, metadata={"help": "If False, masks out inputs in loss."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bf16: bool = field(default=False, metadata={ "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda). This is an experimental API and it may change."})
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"})




def inference(options):
    print(options)
    prompter = Prompter(options.prompt_template_name)
    model_name = get_model_name(options.model_name)
    device_setting = "auto"
    model_class, tokenizer_class, _ = get_model_tokenizer(model_name)
    tokenizer = tokenizer_class.from_pretrained(
        options.model_name_or_path, 
        trust_remote_code=options.trust_remote_code,
    )  

    compute_dtype = (torch.float16 if options.fp16 else (torch.bfloat16 if options.bf16 else torch.float32))        
    model = model_class.from_pretrained(
        options.model_name_or_path,
        load_in_4bit=options.bits == 4,
        load_in_8bit=options.bits == 8,
        device_map=device_setting,
        offload_folder="./cache",    # If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        quantization_config=BitsAndBytesConfig(     
            load_in_4bit=options.bits == 4,
            load_in_8bit=options.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=options.double_quant,
            bnb_4bit_quant_type=options.quant_type,
        ),
        torch_dtype=compute_dtype,
        trust_remote_code=options.trust_remote_code,
    )
    model.config.torch_dtype=compute_dtype
    if options.lora_weights is not None:
        if options.bits <= 8:
            config = LoraConfig.from_pretrained(options.lora_weights)
            model = get_peft_model(model, config) 
            adapters_weights = torch.load(os.path.join(options.lora_weights, WEIGHTS_NAME), map_location=model.device)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            model = PeftModel.from_pretrained(
                model,
                options.lora_weights,
            )
    

    if model_name == "llama":
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    elif model_name == "moss":
        model.config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")


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
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                **kwargs,
            )
        generation_output = generation_output.sequences[0]
        output = tokenizer.decode(generation_output, skip_special_tokens=True)
        output = prompter.get_response(output)
        return output

    records = []
    with open(options.input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            records.append(data)

    with open(options.output_file, "w") as writer:
        for i, record in enumerate(records):
            result = evaluate(instruction=record['instruction'], input=record['input'])
            print(i, result)
            record['output'] = result
            writer.write(json.dumps(record, ensure_ascii=False)+'\n') 



def main():
    hfparser = HfArgumentParser((InferArguments, ))
    options, extra_args =  hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    inference(options)
 


if __name__ == "__main__":
    main()
