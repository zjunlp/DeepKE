from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertForTokenClassification, BertTokenizer, WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
import hydra
from hydra import utils
from deepke.name_entity_re.standard import *

import wandb


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class TrainNer(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None,device=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

wandb.init(project="DeepKE_NER_Standard")
@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    
    # Use gpu or not
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')

    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(cfg.gradient_accumulation_steps))

    cfg.train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if not cfg.do_train and not cfg.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Checkpoints
    if os.path.exists(utils.get_original_cwd()+'/'+cfg.output_dir) and os.listdir(utils.get_original_cwd()+'/'+cfg.output_dir) and cfg.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(utils.get_original_cwd()+'/'+cfg.output_dir))
    if not os.path.exists(utils.get_original_cwd()+'/'+cfg.output_dir):
        os.makedirs(utils.get_original_cwd()+'/'+cfg.output_dir)

    # Preprocess the input dataset
    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    
    # Prepare the model
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model, do_lower_case=cfg.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if cfg.do_train:
        train_examples = processor.get_train_examples(utils.get_original_cwd()+'/'+cfg.data_dir)
        num_train_optimization_steps = int(len(train_examples) / cfg.train_batch_size / cfg.gradient_accumulation_steps) * cfg.num_train_epochs

    config = BertConfig.from_pretrained(cfg.bert_model, num_labels=num_labels, finetuning_task=cfg.task_name)
    model = TrainNer.from_pretrained(cfg.bert_model,from_tf = False,config = config)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(cfg.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list,1)}
    if cfg.do_train:
        train_features = convert_examples_to_features(train_examples, label_list, cfg.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        train_sampler = RandomSampler(train_data)
        
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=cfg.train_batch_size)

        model.train()

        for _ in trange(int(cfg.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask,device)
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps
    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
            wandb.log({
                "train_loss":tr_loss/nb_tr_steps
            })
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(utils.get_original_cwd()+'/'+cfg.output_dir)
        tokenizer.save_pretrained(utils.get_original_cwd()+'/'+cfg.output_dir)
        label_map = {i : label for i, label in enumerate(label_list,1)}
        model_config = {"bert_model":cfg.bert_model,"do_lower":cfg.do_lower_case,"max_seq_length":cfg.max_seq_length,"num_labels":len(label_list)+1,"label_map":label_map}
        json.dump(model_config,open(os.path.join(utils.get_original_cwd()+'/'+cfg.output_dir,"model_config.json"),"w"))
        # Load a trained model and config that you have fine-tuned
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = TrainNer.from_pretrained(utils.get_original_cwd()+'/'+cfg.output_dir)
        tokenizer = BertTokenizer.from_pretrained(utils.get_original_cwd()+'/'+cfg.output_dir, do_lower_case=cfg.do_lower_case)

    model.to(device)

    if cfg.do_eval:
        if cfg.eval_on == "dev":
            eval_examples = processor.get_dev_examples(utils.get_original_cwd()+'/'+cfg.data_dir)
        elif cfg.eval_on == "test":
            eval_examples = processor.get_test_examples(utils.get_original_cwd()+'/'+cfg.data_dir)
        else:
            raise ValueError("eval on dev or test set only")
        eval_features = convert_examples_to_features(eval_examples, label_list, cfg.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=cfg.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask,device=device)

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

        report = classification_report(y_true, y_pred,digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(utils.get_original_cwd()+'/'+cfg.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)

if __name__ == '__main__':
    main()
