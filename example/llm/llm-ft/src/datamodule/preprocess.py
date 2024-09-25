import os
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Tuple, Union

from datasets import load_from_disk

from datamodule.template import get_template_and_fix_tokenizer
from utils.logging import get_logger
from utils.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers.tokenization_utils import PreTrainedTokenizer
    from args import DataArguments, TrainingArguments


logger = get_logger(__name__)


def infer_max_len(source_len: int, target_len: int, data_args: "DataArguments") -> Tuple[int, int]:
    max_target_len = int(data_args.cutoff_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, data_args.max_target_length)
    max_source_len = data_args.cutoff_len - max_target_len
    return max_source_len, max_target_len


def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    for i in range(len(examples["prompt"])):
        query, response = examples["prompt"][i], examples["response"][i]
        query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
        history = examples["history"][i] if "history" in examples else None
        system = examples["system"][i] if "system" in examples else None
        yield query, response, history, system


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    stage: Literal["sft", "sft-custom"]
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                source_len, target_len = len(source_ids), len(target_ids)
                max_source_len, max_target_len = infer_max_len(source_len, target_len, data_args)
                if source_len > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if target_len > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
        
        return model_inputs


    def preprocess_supervised_dataset_seq2seq(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                source_len, target_len = len(source_ids), len(target_ids)
                if source_len > data_args.max_source_length:
                    source_ids = source_ids[:data_args.max_source_length]
                if target_len > data_args.max_target_length:
                    target_ids = target_ids[:data_args.max_target_length]

                input_ids += source_ids 
                labels += target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
        
        return model_inputs
        

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and query != ""):
                continue

            input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

            if template.efficient_eos:
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def print_supervised_dataset_example(example: Dict[str, List[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))


    def print_unsupervised_dataset_example(example: Dict[str, List[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if (stage == "sft" or stage == "sft-custom") and not training_args.predict_with_generate:
        preprocess_func = preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "sft-seq2seq" and not training_args.predict_with_generate:
        preprocess_func = preprocess_supervised_dataset_seq2seq
        print_function = print_supervised_dataset_example
    else:
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    if data_args.cache_path is not None and os.path.exists(data_args.cache_path):
        logger.warning("Loading dataset from disk will ignore other data arguments.")
        return load_from_disk(data_args.cache_path)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=column_names,
            num_proc=data_args.preprocessing_num_workers, 
            **kwargs
        )

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_path`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
