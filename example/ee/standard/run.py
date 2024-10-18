""" BERT-CRF running for ACE 2005 & DuEE1.0 """

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


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bertcrf": (BertConfig, BertCRFForTokenClassification, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, eval_dataset, model, tokenizer, labels, pad_token_label_id, device):
    """ Train the model """
    OmegaConf.set_struct(args, True)
    with open_dict(args):
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
    )
    best_dev_f1=0.0
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    # for crf might wanna change this to SGD with gradient clipping and my other scheduler **** eskiler adamken bile 1000lerden başlayıp 10lara düşüyo
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[4] if args.model_type in ["bert", "bertcrf",
                                                                           "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                #optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, eval_dataset, tokenizer, labels, pad_token_label_id, mode="dev", device=device)
                        if results['f1']>best_dev_f1:
                            best_dev_f1=results['f1']

                            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                                # Save model checkpoint
                                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training

                                logger.info("Saving model checkpoint to %s", output_dir)

                                model_save_path_ = os.path.join(output_dir, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(), model_save_path_)
                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                logger.info("Saving model checkpoint to %s", output_dir)

                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    #/if args.local_rank in [-1, 0]:
        #tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, tokenizer, labels, pad_token_label_id, mode, device, prefix=""):

    OmegaConf.set_struct(args, True)
    with open_dict(args):
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Mode = %s", mode)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[4] if args.model_type in ["bert", "bertcrf",
                                                                           "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            tmp_eval_loss, logits, best_path = outputs

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        if preds is None:
            preds = best_path.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, best_path.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list

def load_and_cache_examples(args, examples, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir, 
        "cached_{}_{}_{}".format(mode,
        list(filter(None,args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length))
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_examples_to_features(
            examples, 
            labels, 
            args.max_seq_length, tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),# xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),# roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]), # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_token_type_ids = None if features[0].token_type_ids is None else torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_token_type_ids)
    return dataset

@hydra.main(config_path="./conf", config_name="train.yaml")
def main(args):
    
    cwd = utils.get_original_cwd()
    OmegaConf.set_struct(args, True)
    with open_dict(args):
        args["cwd"] = cwd
    args.data_dir = os.path.join(args.cwd, "./data/" + args.data_name + "/" + args.task_name)
    args.tag_path = os.path.join(args.cwd, "./data/" + args.data_name + "/schema")
    args.output_dir = os.path.join(args.cwd, "./exp/" + args.data_name + "/" + args.task_name + "/" + args.model_name_or_path.split("/")[-1])
    # args.model_name_or_path = os.path.join(args.cwd, args.model_name_or_path)
    args.do_predict = True if args.data_name == "ACE" else False



    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

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

    # OmegaConf.set_struct(args, True)
    # with open_dict(args):
    #     print(device)
    #     print(device.type)
    #     args["device"] = device

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
    # tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    pad_token_label_id = -100
    # get_processor
    processor = PROCESSORS[args.data_name](task_name=args.task_name, tokenizer=tokenizer)

    labels = processor.get_labels(args.tag_path)
    num_labels = len(labels)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # config
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    # load model
    ner_model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    ner_model.to(device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_examples = processor.get_examples(os.path.join(args.data_dir, "train.tsv"), "train")
        train_dataset = load_and_cache_examples(args, train_examples, tokenizer, labels, pad_token_label_id, mode="train")

    if args.do_eval:
        eval_examples = processor.get_examples(os.path.join(args.data_dir, "dev.tsv"), "dev")
        eval_dataset = load_and_cache_examples(args, eval_examples , tokenizer, labels, pad_token_label_id, mode="dev")

    if args.do_predict:    
        test_examples = processor.get_examples(os.path.join(args.data_dir, "test.tsv"), "test")
        test_dataset = load_and_cache_examples(args, test_examples , tokenizer, labels, pad_token_label_id, mode="test")

    # Training
    if args.do_train:
        global_step, tr_loss = train(
            args=args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            model=ner_model, 
            tokenizer=tokenizer, 
            labels=labels, 
            pad_token_label_id=pad_token_label_id,
            device=device)
        print(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

     # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            ner_model.module if hasattr(ner_model, "module") else ner_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            result, eval_pred_list = evaluate(args, model, eval_dataset, tokenizer, labels, pad_token_label_id, mode="dev", device=device, prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        output_eval_pred_file = os.path.join(args.output_dir, "eval_pred.json")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
        
        # dump the trigger pred result of dev.
        json.dump(eval_pred_list, open(output_eval_pred_file, "w"), ensure_ascii=False)


    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(device)
        result, test_pred_list = evaluate(args, model, test_dataset, tokenizer, labels, pad_token_label_id, mode="test", device=device)
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_pred_file = os.path.join(args.output_dir, "test_pred.json")
        json.dump(test_pred_list, open(output_test_pred_file, "w"), ensure_ascii=False)

    return results


if __name__ == "__main__":
    main()