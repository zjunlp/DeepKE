from argparse import ArgumentParser
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from opendelta import LoraModel
import torch
import json
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--use-bminf", default=False, action="store_true", help="Whether to use BMInf")
    parser.add_argument("--memory-limit", type=int, default=20, help="GPU Memory limit, in GB")
    parser.add_argument("--delta", default=None, type=str, help="The path to lora.")
    parser.add_argument("--device", default="cuda:0", type=str, help="The target device.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_list=[]
    with open('valid1.json', 'r', encoding='utf-8') as f:
        d = [json.loads(line.strip()) for line in f]
        for data in d:
            data_list.append(
                {"input": data["input"], "prompt": data["instruction"], "<ans>": ""})

    config = CPMBeeConfig.from_json_file("config/cpm-bee-10b.json")
    ckpt_path = "path/to/checkpoint.pt"
    tokenizer = CPMBeeTokenizer()
    model = CPMBeeTorch(config=config)

    if args.delta is not None:
        delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"], backend="hf")
        model.load_state_dict(torch.load(args.delta), strict=False)

    model.load_state_dict(torch.load(ckpt_path), strict=False)

    if args.device == "cpu":
        model = model.float()
    else:
        if not torch.cuda.is_available():
            raise AssertionError("The CUDA is unavailable")
        if args.use_bminf:
            import bminf
            with torch.cuda.device(args.device):
                model = bminf.wrapper(model, quantization=False, memory_limit=args.memory_limit << 30)
        model.cuda(args.device)

    # use beam search
    beam_search = CPMBeeBeamSearch(
        model=model,
        tokenizer=tokenizer,
    )
    inference_results = beam_search.generate(data_list, max_length=100, repetition_penalty=1.1)
    with open('cpm_bee_TG.json', 'w', encoding='utf-8') as fw_ans:
        for res in inference_results:
            fw_ans.write(res+"\n")

if __name__ == "__main__":
    main()
