import argparse
import os
import json
import torch
from datasets import Dataset
import time
from tqdm import tqdm

from args.parser import get_infer_args
from datamodule.preprocess import preprocess_dataset
from utils.general_utils import get_model_tokenizer_trainer, get_model_name
from utils.logging import get_logger
from utils.load_cmd import load_config_from_yaml

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

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

logger = get_logger(__name__)


def convert_datas2datasets(records):
    dataset_mapping = {'prompt':[], 'query':[], "response":[]}
    for record in records:
        dataset_mapping['prompt'].append(record['instruction'])
        dataset_mapping['query'].append(record.get('input', ''))
        dataset_mapping['response'].append(record.get('output', 'test'))
    dataset = Dataset.from_dict(dataset_mapping)
    return dataset


def load_samplingparam(generating_args, inference_args):
    if 'greedy' == inference_args.gen_mode:
        if generating_args.num_beams != 1:
            sampling_params = SamplingParams(
                n=generating_args.num_beams,
                best_of=generating_args.num_beams,
                temperature=0.2,
                use_beam_search=False,
                max_tokens=generating_args.max_new_tokens,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.2,
                use_beam_search=False,
                max_tokens=generating_args.max_new_tokens,
            )
    elif "speculative_sample" == inference_args.gen_mode:
        sampling_params = SamplingParams(
            use_speculative_sampling=True, max_tokens=generating_args.max_new_tokens
        )
    else:
        sampling_params = None
    return sampling_params



def inference(model_args, data_args, training_args, finetuning_args, generating_args, inference_args):
    model_class, tokenizer_class, _ = get_model_tokenizer_trainer(model_args.model_name)
    logger.info(f"model_class:{model_class}\ntokenizer_class:{tokenizer_class}\n")
    tokenizer = tokenizer_class.from_pretrained(
        model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens, padding_side="left",
        trust_remote_code=True, cache_dir=model_args.cache_dir, revision=model_args.model_revision,
    )

    llm = LLM(model=model_args.model_name_or_path, trust_remote_code=True)
    sampling_params = load_samplingparam(generating_args, inference_args)


    os.makedirs(inference_args.output_file, exist_ok=True)
    if inference_args.process_mode == 'file':
        input_files = inference_args.input_file.split(',')
        for input_file in input_files:
            output_file = os.path.join(inference_args.output_file, input_file.split('/')[-1])
            records = []
            with open(input_file, "r") as reader:
                for line in reader:
                    data = json.loads(line)
                    records.append(data)
            if len(records) == 0:
                    continue
            predict_dataset = convert_datas2datasets(records)
            predict_dataset = preprocess_dataset(predict_dataset, tokenizer, data_args, training_args, finetuning_args.stage)

            outputs = llm.generate(
                sampling_params=sampling_params,
                prompt_token_ids=predict_dataset["input_ids"],
                use_tqdm=True,
            )

            with open(output_file, inference_args.mode) as writer:
                for record, output in zip(records, outputs):
                    record['output'] = output.outputs[0].text
                    writer.write(json.dumps(record, ensure_ascii=False)+'\n')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file")
    parser.add_argument('--stage', type=str, default='sft', help="Stage of the model")
    parser.add_argument('--model_name_or_path', type=str, default='lora_results/baichuan2-13b-v1/baichuan2-13b-v1', help="Path to model or model name")
    parser.add_argument('--model_name', type=str, default='baichuan', help="Model name")
    parser.add_argument('--template', type=str, default='baichuan2', help="Template name")
    parser.add_argument('--do_predict', action='store_true', default=True, help="Whether to perform prediction")
    parser.add_argument('--input_file', type=str, default='data/input.json', help="Path to the input file")
    parser.add_argument('--output_file', type=str, default='results/baichuan2-13b-IEPile-lora_output.json', help="Path to the output file")
    parser.add_argument('--output_dir', type=str, default='lora_results/test', help="Directory to save output")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for inference")
    parser.add_argument('--predict_with_generate', action='store_true', default=True, help="Whether to use generation during inference")
    parser.add_argument('--max_source_length', type=int, default=1024, help="Maximum source length")
    parser.add_argument('--bf16', action='store_true', default=True, help="Whether to use bf16")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    if args.config:
        #  if user provided a config file, load it
        config = load_config_from_yaml(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


def main(args=None):
    args = set_args()

    model_args, data_args, training_args, finetuning_args, generating_args, inference_args = get_infer_args(args)

    # model_name映射
    model_args.model_name = get_model_name(model_args.model_name)
    inference(model_args, data_args, training_args, finetuning_args, generating_args, inference_args)


if __name__ == "__main__":
    main()
