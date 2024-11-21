import argparse
import os
import sys
sys.path.append('./')
import json
import torch
from datasets import Dataset
from transformers import GenerationConfig
from tqdm import tqdm

from args.parser import get_infer_args
from model.loader import load_model_and_tokenizer
from datamodule.preprocess import preprocess_dataset
from utils.general_utils import get_model_tokenizer_trainer, get_model_name
from utils.logging import get_logger
from utils.load_cmd import load_config_from_yaml


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
        dataset_mapping['query'].append(record.get('input', None))
        dataset_mapping['response'].append(record.get('output', 'test'))
    dataset = Dataset.from_dict(dataset_mapping)
    return dataset


def inference(model_args, data_args, training_args, finetuning_args, generating_args, inference_args):
    model_class, tokenizer_class, _ = get_model_tokenizer_trainer(model_args.model_name)
    logger.info(f"model_class:{model_class}\ntokenizer_class:{tokenizer_class}\n")
    model, tokenizer = load_model_and_tokenizer(
        model_class,
        tokenizer_class,
        model_args,
        finetuning_args,
        training_args.do_train,
        stage="sft",
    )
    if model_args.bits >= 8:
        model.to(device)
    print(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")


    def evaluate(
        input_ids,
        generating_args,
        **kwargs,
    ):
        input_length = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        input_ids = input_ids.to(device)
        generation_config = GenerationConfig(
            temperature=generating_args.temperature,
            top_p=generating_args.top_p,
            top_k=generating_args.top_k,
            num_beams=generating_args.num_beams,
            max_new_tokens=generating_args.max_new_tokens,
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
        generation_output = generation_output[input_length:]
        output = tokenizer.decode(generation_output, skip_special_tokens=True)
        return output



    records = []
    with open(inference_args.input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            data["output"] = "test"
            records.append(data)
    predict_dataset = convert_datas2datasets(records)
    predict_dataset = preprocess_dataset(predict_dataset, tokenizer, data_args, training_args, finetuning_args.stage)


    with open(inference_args.output_file, 'w') as writer:
        for record, model_inputs in zip(records, predict_dataset["input_ids"]):
            result = evaluate(model_inputs, generating_args)
            print(result)
            record['output'] = result
            writer.write(json.dumps(record, ensure_ascii=False)+'\n')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file")
    parser.add_argument('--stage', type=str, default='sft', help="Stage of the model")
    parser.add_argument('--model_name_or_path', type=str, default='models/llama2-13B-Chat', help="Path to model or model name")
    parser.add_argument('--checkpoint_dir', type=str, default='lora/llama2-13b-IEPile-lora', help="Directory of the checkpoint")
    parser.add_argument('--model_name', type=str, default='llama', help="Model name")
    parser.add_argument('--template', type=str, default='llama2', help="Template for the model")
    parser.add_argument('--do_predict', action='store_true', default=True, help="Whether to do prediction")
    parser.add_argument('--input_file', type=str, default='data/input.json', help="Path to the input file")
    parser.add_argument('--output_file', type=str, default='results/llama2-13b-IEPile-lora_output.json', help="Path to the output file")
    parser.add_argument('--finetuning_type', type=str, default='lora', help="Finetuning type (e.g., 'lora')")
    parser.add_argument('--output_dir', type=str, default='lora/test', help="Directory to store output")
    parser.add_argument('--predict_with_generate', action='store_true', default=True, help="Whether to predict with generate")
    parser.add_argument('--cutoff_len', type=int, default=512, help="Max length for the input")
    parser.add_argument('--bf16', action='store_true', default=True, help="Whether to use bf16 precision")
    parser.add_argument('--max_new_tokens', type=int, default=300, help="Maximum number of new tokens to generate")
    parser.add_argument('--bits', type=int, default=4, help="Bits for quantization")
    args = parser.parse_args()

    if args.config:
        # if user provides a config file, load the config from the file
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
