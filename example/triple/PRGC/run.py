# /usr/bin/env python
# coding=utf-8
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import hydra
from hydra.utils import get_original_cwd
import torch
from transformers import BertConfig
import random
import logging
from tqdm import trange
import argparse
import wandb


# import utils
# from optimization import BertAdam
# from evaluate import evaluate
# from dataloader import CustomDataLoader
# from model import BertForRE

from deepke.triple_extraction.PRGC import *

def train(model, data_iterator, optimizer, params, ex_params):
    """Train the model one epoch
    """
    # set model to training mode
    model.train()

    loss_avg = util.RunningAverage()
    loss_avg_seq = util.RunningAverage()
    loss_avg_mat = util.RunningAverage()
    loss_avg_rel = util.RunningAverage()

    # Use tqdm for progress bar
    # one epoch
    t = trange(len(data_iterator), ascii=True)
    for step, _ in enumerate(t):
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batch

        # compute model output and loss
        loss, loss_seq, loss_mat, loss_rel = model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                                   potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                                   ex_params=ex_params)

        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps

        # back-prop
        loss.backward()

        if (step + 1) % params.gradient_accumulation_steps == 0:
            # performs updates using calculated gradients
            optimizer.step()
            model.zero_grad()

        # update the average loss
        loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        loss_avg_seq.update(loss_seq.item())
        loss_avg_mat.update(loss_mat.item())
        loss_avg_rel.update(loss_rel.item())
        # 右边第一个0为填充数，第二个5为数字个数为5位，第三个3为小数点有效数为3，最后一个f为数据类型为float类型。
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()),
                      loss_seq='{:05.3f}'.format(loss_avg_seq()),
                      loss_mat='{:05.3f}'.format(loss_avg_mat()),
                      loss_rel='{:05.3f}'.format(loss_avg_rel()))


def train_and_evaluate(model, params, ex_params, restore_file=''):
    """Train the model and evaluate every epoch."""
    # Load training data and val data
    dataloader = CustomDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train', ex_params=ex_params)
    val_loader = dataloader.get_dataloader(data_sign='val', ex_params=ex_params)
    # reload weights from restore_file if specified
    if restore_file != '':
        restore_path = os.path.join(params.model_dir, params.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        # 读取checkpoint
        model, optimizer = util.load_checkpoint(restore_path)

    model.to(params.device)
    # wandb.watch(model, log="all")
    # parallel model
    if params.n_gpu > 1 and params.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    # fine-tuning
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        # pretrain model param
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.fin_tuning_lr
         },
        # downstream model
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.downs_en_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.downs_en_lr
         }
    ]
    num_train_optimization_steps = len(train_loader) // params.gradient_accumulation_steps * params.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)

    # patience stage
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Train for one epoch on training set
        train(model, train_loader, optimizer, params, ex_params)

        # Evaluate for one epoch on training set and validation set
        # train_metrics = evaluate(args, model, train_loader, params, mark='Train',
        #                          verbose=True)  # Dict['loss', 'f1']
        val_metrics, _, _ = evaluate(model, val_loader, params, ex_params, mark='Val')
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        util.save_checkpoint({'epoch': epoch + 1,
                               'model': model_to_save,
                               'optim': optimizer_to_save},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(os.path.join(params.ex_dir, 'params.json'))

        # stop training based params.patience
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break
        
        wandb.log({
            "best_val_f1":best_val_f1,
        })
        


wandb.init(project="DeepKE_TRIPLE_PRGC")
wandb.watch_called = False
@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    cwd = get_original_cwd()
    os.chdir(cwd)
    params = util.Params(cfg.ex_index, cfg.corpus_type)
    setattr(params, 'root_path', cwd)
    setattr(params, 'data_dir', os.path.join(params.root_path, f'data/{cfg.corpus_type}'))
    setattr(params, 'ex_dir', os.path.join(params.root_path, f'experiments/ex{cfg.ex_index}'))
    setattr(params, 'model_dir', os.path.join(params.root_path, f'model/ex{cfg.ex_index}'))
    setattr(params, 'bert_model_dir', os.path.join(params.root_path, f'pretrain_models', cfg.pretrain_model))
    setattr(params, 'rel2idx', json.load(open(os.path.join(params.data_dir,'rel2id.json'), 'r', encoding='utf-8'))[-1])
    setattr(params, 'rel_num', len(params.rel2idx))
    for key,value in cfg.items():
        setattr(params, key, value)
    
    ex_params = {
        'ensure_corres': cfg.ensure_corres,
        'ensure_rel': cfg.ensure_rel,
        'num_negs': cfg.num_negs,
        'emb_fusion': cfg.emb_fusion
    }

    if params.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        torch.cuda.set_device(params.device_id)
        print('current device:', torch.cuda.current_device())
        params.n_gpu = n_gpu = 1

    # Set the random seed for reproducible experiments
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    params.seed = params.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(params.seed)

    # Set the logger
    util.set_logger(save=True, log_path=os.path.join(params.ex_dir, 'train.log'))
    logging.info(f"Model type:")
    logging.info("device: {}".format(params.device))

    logging.info('Load pre-train model weights...')
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    logging.info('-done')

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, params, ex_params, params.restore_file)
    
if __name__ == "__main__":
    main()