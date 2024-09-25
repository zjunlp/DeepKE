""" BERT-CRF predict for ACE 2005 & DuEE1.0 """

import argparse
import glob
import logging
import os
import sys
import json
import random

import numpy as np
import torch
import hydra
from hydra import utils
from omegaconf import OmegaConf, open_dict
from seqeval.metrics import f1_score, precision_score, recall_score

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from deepke.event_extraction.standard.bertcrf.processor_ee import convert_examples_to_features, PROCESSORS
from deepke.event_extraction.standard.bertcrf.bert_crf import *
from run import evaluate, load_and_cache_examples, set_seed

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bertcrf": (BertConfig, BertCRFForTokenClassification, BertTokenizer),
}

@hydra.main(config_path="./conf", config_name="predict.yaml")
def main(args):
    cwd = utils.get_original_cwd()
    OmegaConf.set_struct(args, True)
    with open_dict(args):
        args["cwd"] = cwd
    args.data_dir = os.path.join(args.cwd, "./data/" + args.data_name + "/" + args.task_name)
    args.tag_path = os.path.join(args.cwd, "./data/" + args.data_name + "/schema")
    args.model_name_or_path = os.path.join(args.cwd, args.model_name_or_path)
    args.dev_trigger_pred_file = os.path.join(args.cwd, args.dev_trigger_pred_file) if args.do_pipeline_predict and args.task_name=="role" else None
    args.test_trigger_pred_file = os.path.join(args.cwd, args.test_trigger_pred_file) if args.do_pipeline_predict and args.task_name=="role" else None
    args.do_predict = True if args.data_name == "ACE" else False

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info({"n_gpu: ": args.n_gpu})

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    logger.info(f"label_nums:{config.num_labels}")
    model.to(device)

    pad_token_label_id = -100
    # get_processor
    processor = PROCESSORS[args.data_name](task_name=args.task_name, tokenizer=tokenizer)
    labels = processor.get_labels(args.tag_path)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    logger.info("Training/evaluation parameters %s", args)

    raw_path = "/".join(args.data_dir.split("/")[:-1])
    if args.do_eval:
        if args.task_name=="role" and args.dev_trigger_pred_file is not None:
            processor.process_dev_with_pred_trigger(args, raw_path, "dev_with_pred_trigger.tsv")
            eval_examples = processor.get_examples(os.path.join(args.data_dir, "dev_with_pred_trigger.tsv"), "dev")
        else:
            eval_examples = processor.get_examples(os.path.join(args.data_dir, "dev.tsv"), "dev")
        eval_dataset = load_and_cache_examples(args, eval_examples , tokenizer, labels, pad_token_label_id, mode="dev")

    if args.do_predict:
        if args.task_name=="role" and args.test_trigger_pred_file is not None:
            processor.process_test_with_pred_trigger(args, raw_path, "test_with_pred_trigger.tsv")
            test_examples = processor.get_examples(os.path.join(args.data_dir, "test_with_pred_trigger.tsv"), "test")
        else:
            test_examples = processor.get_examples(os.path.join(args.data_dir, "test.tsv"), "test")
        test_dataset = load_and_cache_examples(args, test_examples , tokenizer, labels, pad_token_label_id, mode="test")

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:      

        result, eval_pred_list = evaluate(args, model, eval_dataset, tokenizer, labels, pad_token_label_id, mode="dev", device=device)
        results.update(result)
        output_eval_file = os.path.join(args.model_name_or_path, "eval_results.txt")
        output_eval_pred_file = os.path.join(args.model_name_or_path, "eval_pred.json")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
        
        # dump the trigger pred result of dev.
        json.dump(eval_pred_list, open(output_eval_pred_file, "w"), ensure_ascii=False)

    if args.do_predict and args.local_rank in [-1, 0]:
        
        result, test_pred_list = evaluate(args, model, test_dataset, tokenizer, labels, pad_token_label_id, mode="test", device=device)
        # Save results
        output_test_results_file = os.path.join(args.model_name_or_path, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_pred_file = os.path.join(args.model_name_or_path, "test_pred.json")
        json.dump(test_pred_list, open(output_test_pred_file, "w"), ensure_ascii=False)

    return results

if __name__ == "__main__":
    main()