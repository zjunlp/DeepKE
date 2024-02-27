import os
import sys
sys.path.append('./')
import json
import torch
from datasets import Dataset
from transformers import GenerationConfig

from args.parser import get_infer_args
from model.loader import load_model_and_tokenizer
from datamodule.preprocess import preprocess_dataset
from utils.general_utils import get_model_tokenizer_trainer, get_model_name
from utils.logging import get_logger
from tqdm import tqdm

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



def main(args=None):
    model_args, data_args, training_args, finetuning_args, generating_args, inference_args = get_infer_args(args)
    # model_name映射
    model_args.model_name = get_model_name(model_args.model_name)
    inference(model_args, data_args, training_args, finetuning_args, generating_args, inference_args)
 



if __name__ == "__main__":
    main()
