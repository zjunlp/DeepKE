"""Evaluate the model"""
import json
import logging
import random
import argparse

from tqdm import tqdm
import os

import torch
import numpy as np
import pandas as pd

import hydra
from hydra.utils import get_original_cwd

# from metrics import tag_mapping_nearest, tag_mapping_corres
# from util import Label2IdxSub, Label2IdxObj
# import util
# from dataloader import CustomDataLoader
from deepke.triple_extraction.PRGC import *

def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output


def evaluate(model, data_iterator, params, ex_params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    rel_num = params.rel_num

    predictions = []
    ground_truths = []
    correct_num, predict_num, gold_num = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)

            # (sum(x_i), seq_len)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            # (bs, seq_len, seq_len)
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            # (bs,)
            xi = np.array(xi)
            # (sum(s_i),)
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            xi_index.insert(0, 0)

        for idx in range(bs):
            if ex_params['ensure_rel']:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)

            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])
            ground_truths.append(list(set(gold_triples)))
            predictions.append(list(set(pre_triples)))
            # counter
            correct_num += len(set(pre_triples) & set(gold_triples))
            predict_num += len(set(pre_triples))
            gold_num += len(set(gold_triples))
    metrics = get_metrics(correct_num, predict_num, gold_num)
    # logging loss, f1 and report
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics:\n".format(mark) + metrics_str)
    return metrics, predictions, ground_truths


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
    setattr(params, 'restore_file', 'last')
    
    ex_params = {
        'corres_threshold': params.corres_threshold,
        'rel_threshold': params.rel_threshold,
        'ensure_corres': params.ensure_corres,
        'ensure_rel': params.ensure_rel,
        'emb_fusion': params.emb_fusion
    }

    torch.cuda.set_device(params.device_id)
    print('current device:', torch.cuda.current_device())
    mode = cfg.mode
    # Set the random seed for reproducible experiments
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    params.seed = params.seed

    # Set the logger
    util.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, params.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = util.load_checkpoint(os.path.join(params.model_dir, params.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')

    logging.info("Starting prediction...")
    _, predictions, ground_truths = evaluate(model, loader, params, ex_params, mark=mode)
    with open(os.path.join(params.data_dir , f'{mode}_triples.json'), 'r', encoding='utf-8') as f_src:
        src = json.load(f_src)
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'pre': predictions,
                'truth': ground_truths
            }
        )
        df.to_csv(os.path.join(params.ex_dir , f'{mode}_result.csv'))
    logging.info('-done')

if __name__ == "__main__":
    main()