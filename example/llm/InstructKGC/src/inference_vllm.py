import os
import json
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from peft import PeftModel
from vllm.sampling_params import SamplingParams

from utils.prompter import Prompter
from utils.general_utils import get_model_tokenizer, get_model_name
from utils.vllm_llm import LLM


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
    mode: str = field(default='w')
    model_name_or_path: str = field(default=None, metadata={"help": "Model name or path."})
    model_name: str = field(default='llama', metadata={"help": "all supported models can be found in ./src/utils/general_utils.py:MODEL_DICT."})
    lora_weights: str = field(default=None, metadata={"help": "Path to the lora weights directory."})
    input_file: str = field(default=None, metadata={"help": "Path to the input file."})
    output_file: str = field(default=None, metadata={"help": "Path to the output file."})
    prompt_template_name: str = field(default='alpaca', metadata={"help": "Prompt template name. All prompt template can be found in ./templates"})
    id_text: str = field(default='input')

    gen_mode: str = field(default='greedy', metadata={"help": "gen_mode."})
    max_source_length: int = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    max_target_length: int = field(default=256, metadata={"help": "The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    temperature: float = field(default=0.2)
    top_k: int = field(default=40)
    top_p: float = field(default=0.75)
    repetition_penalty: float = field(default=1.3)
    num_beams: int = field(default=4)
    max_new_tokens: int = field(default=256)
    batch_size: int = field(default=16)
    
    trust_remote_code: bool = field(default=True, metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})
    train_on_inputs: bool = field(default=False, metadata={"help": "If False, masks out inputs in loss."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bf16: bool = field(default=False, metadata={ "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda). This is an experimental API and it may change."})
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"})


def load_llm(options, state_dict):
    if 'greedy' == options.gen_mode:
        sampling_params = SamplingParams(
        temperature=0.0,
        use_beam_search=False,
        max_tokens=options.max_new_tokens,
    )
    elif "speculative_sample" == options.gen_mode:
        sampling_params = SamplingParams(use_speculative_sampling=True, max_tokens=options.max_new_tokens)
    else:
        sampling_params = None

    llm = LLM(
        model = options.model_name_or_path,
        tokenizer = options.model_name_or_path,
        state_dict = state_dict,
        batch_size = options.batch_size,
    )
    return llm, sampling_params


def merge_model(options, base_model):
    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()
    
    lora_model = PeftModel.from_pretrained(
        base_model,
        options.lora_weights,
    )
    
    assert torch.allclose(first_weight_old, first_weight)

    print("Merging . . .")
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    assert not torch.allclose(first_weight_old, first_weight)
    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }
    return deloreanized_sd


def inference(options):
    print(options)
    prompter = Prompter(options.prompt_template_name)
    model_name = get_model_name(options.model_name)
    device_setting = {"": 0} if model_name == "cpm-bee" else "auto"
    model_class, tokenizer_class, _ = get_model_tokenizer(model_name)


    tokenizer = tokenizer_class.from_pretrained(
        options.model_name_or_path, 
        trust_remote_code=options.trust_remote_code,
    )  
    compute_dtype = (torch.float16 if options.fp16 else (torch.bfloat16 if options.bf16 else torch.float32))  
    base_model = model_class.from_pretrained(
        options.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=compute_dtype,
        device_map=device_setting,
    )


    if model_name == "llama":
        base_model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        base_model.config.bos_token_id = tokenizer.bos_token_id = 1
        base_model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    elif model_name == "moss":
        base_model.config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")


    state_dict = None
    if options.lora_weights is not None:
        state_dict = merge_model(options, base_model)
    else:
        state_dict = base_model.state_dict()
    llm, sampling_params = load_llm(options, state_dict)


    already = set()
    if options.mode == "a":
        with open(options.output_file, "r") as reader:
            for line in reader:
                data = json.loads(line)
                already.add(data[options.id_text])


    records = []
    prompt_token_ids = []
    with open(options.input_file, "r") as reader:
        for line in reader:
            record = json.loads(line)
            if record[options.id_text] in already:
                print(f'{record[options.id_text]} has already exists!')
                continue
            records.append(record)
            prompt_ = prompter.generate_prompt(instruction=record['instruction'], input=record.get('input', None))
            inputs = tokenizer(prompt_)
            input_ids = inputs["input_ids"][:options.max_source_length]
            prompt_token_ids.append(input_ids)


    outputs = llm.generate(
        sampling_params=sampling_params,
        prompt_token_ids=prompt_token_ids, 
        use_tqdm=True,
    )
    
    
    with open(options.output_file, options.mode) as writer:
        for record, output in zip(records, outputs):
            record['output'] = output.outputs[0].text
            writer.write(json.dumps(record, ensure_ascii=False)+'\n') 


def main():
    hfparser = HfArgumentParser((InferArguments, ))
    options, extra_args =  hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    inference(options)
 




if __name__ == "__main__":
    main()
