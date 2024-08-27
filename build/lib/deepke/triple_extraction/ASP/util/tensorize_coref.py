import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import numpy as np
import random
from transformers import T5Tokenizer

import os
from os.path import join
import json
import pickle
import logging
import torch
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import util

logger = logging.getLogger(__name__)

class CorefDataProcessor:
    def __init__(self, config, language='english'):
        self.config = config
        self.language = language

        self.max_seg_len = config['max_segment_len']

        self.data_dir = config['data_dir']
        self.dataset = config['dataset']

        # Get tensorized samples
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            self.tensor_samples, self.stored_info = pickle.load(open(cache_path, 'rb'))
            logger.info(f'Loaded tensorized examples from cache: {cache_path}')
        else:
            # Generate tensorized samples
            if self.dataset == "ontonotes_coref":
                self.tensor_samples = {}
                tensorizer = Tensorizer(self.config)
                suffix = f'{self.config["plm_tokenizer_name"]}.{language}.{self.max_seg_len}.jsonlines'

                paths = {
                    'train': join(self.data_dir, f'train.{suffix}'),
                    'dev': join(self.data_dir, f'dev.{suffix}'),
                    'test': join(self.data_dir, f'test.{suffix}')
                }
                
            if "t5-small" == self.config["plm_tokenizer_name"]: # use full-size data for evaluation
                suffix = f'{self.config["plm_tokenizer_name"]}.{language}.4096.jsonlines'
                paths['dev'] = join(self.data_dir, f'dev.{suffix}')
                paths['test'] = join(self.data_dir, f'test.{suffix}')
                
            for split, path in paths.items():
                logger.info(f'Tensorizing examples from {path}; results will be cached)')
                is_training = (split == 'train')

                samples = json.load(open(path))
                tensor_samples = [tensorizer.tensorize_example(sample, is_training) for sample in samples]                    

                self.tensor_samples[split] = CorefDataset(
                    sorted(
                        [(doc_key, tensor) for doc_key, tensor in tensor_samples],
                        key=lambda x: -x[1]['input_ids'].size(0)
                    )
                )
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            pickle.dump((self.tensor_samples, self.stored_info), open(cache_path, 'wb'))
            logger.info(f'Cached tensorized examples to {cache_path}')

    def get_tensor_examples(self):
        # For each split, return list of tensorized samples to allow variable length input (batch size = 1)
        return self.tensor_samples['train'], self.tensor_samples['dev'], self.tensor_samples['test']

    def get_stored_info(self):
        return self.stored_info

    def get_cache_path(self):
        if self.dataset == "ontonotes_coref":
            cache_path = join(
                self.data_dir, 
                f'cached.tensors.{self.config["plm_tokenizer_name"]}.{self.max_seg_len}.bin'
            )
        return cache_path


class Tensorizer:
    def __init__(self, config):
        self.config = config
        self.tz = T5Tokenizer.from_pretrained(config['plm_tokenizer_name'])
        
        SPEAKER_START = '<speaker>'
        SPEAKER_END   = '</speaker>'
        MENTION_START = '<m>'
        MENTION_END   = '</m>'

        self.tz.add_tokens(SPEAKER_START)
        self.tz.add_tokens(SPEAKER_END)
        self.tz.add_tokens(MENTION_START)
        self.tz.add_tokens(MENTION_END)

        self.mention_start_id = self.tz.convert_tokens_to_ids(MENTION_START)
        self.mention_end_id   = self.tz.convert_tokens_to_ids(MENTION_END)

        # Will be used in evaluation
        self.stored_info = {
            'example': {},  # {doc_key: ...}
            'subtoken_maps': {},  # {doc_key: ...}
        }


    def get_action_labels(
        self, label_ids
    ):
        # replacing natural language tokens with <copy>: action 0
        # <m> with action 1
        # </m> with action 2 
        action_labels = torch.where(
            label_ids != self.tz.pad_token_id, label_ids, torch.ones_like(
                label_ids)*(-103)
        )
        action_labels = torch.where(
            action_labels == self.mention_start_id, -2, action_labels
        )
        action_labels = torch.where(
            action_labels == self.mention_end_id, -1, action_labels
        )
        action_labels = torch.where(
            (action_labels != -1) & (action_labels != -2) & (action_labels >= 0), 
            -3, action_labels
        )
        action_labels += 3
        return action_labels


    def tensorize_example(
        self, example, is_training
    ):
        # Keep info to store
        doc_key = example['doc_key']
        self.stored_info['example'][doc_key] = example
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        # self.stored_info['gold'][doc_key] = example['clusters']

        is_training = torch.tensor(is_training, dtype=torch.bool)

        # Sentences/segments
        input_sentence = copy.deepcopy(example['sentence'])  # Segments
        target_sentence = copy.deepcopy(example["target_sentence"])

        cluster_category = copy.deepcopy(example['cluster_category'])
        mention_indice = copy.deepcopy(example['mention_indice'])
        
        sentence_map = torch.tensor(example['sentence_map'], dtype=torch.long)
        
        input_ids = torch.tensor(self.tz.convert_tokens_to_ids(input_sentence), dtype=torch.long)
        target_ids = torch.tensor(self.tz.convert_tokens_to_ids(target_sentence), dtype=torch.long)
        input_len, target_len = input_ids.size(0), target_ids.size(0)
        
        input_mask = torch.tensor([1] * input_len, dtype=torch.long)
        target_mask = torch.tensor([1] * target_len, dtype=torch.long)

        action_labels = self.get_action_labels(target_ids)
        mention_indice = torch.tensor(mention_indice, dtype=torch.long)
        cluster_category = torch.tensor(cluster_category, dtype=torch.long)

        linearized_indices = torch.arange(target_len)

        is_l = (target_ids == self.mention_start_id)
        is_r = (target_ids == self.mention_end_id)

        # (num_l)
        l_pos = linearized_indices[is_l]
        # (target_len, num_l)
        distance_to_previous_l = linearized_indices.unsqueeze(1) - l_pos.unsqueeze(0)
        # (target_len, num_l)
        is_after_l = (distance_to_previous_l > 0)
        # (target_len, num_l)
        lr_pair_flag = (
            mention_indice.unsqueeze(1) == l_pos.unsqueeze(0) # paired brackets are true, otherwise false
        ) & is_after_l # keeping right after left only

        # (num_r)
        r_pos = linearized_indices[is_r]
        # (target_len, num_r)
        distance_to_previous_r = linearized_indices.unsqueeze(1) - r_pos.unsqueeze(0)
        # (target_len, num_r)
        is_after_r = (distance_to_previous_r > 0)
        # (target_len, 1) == (1, num_r) -> (target_len, num_r)
        rr_pair_flag = (
            cluster_category.unsqueeze(1) == cluster_category[is_r].unsqueeze(0) # right brackets from the same cluster are true, otherwise false
        ) & is_after_r # keeping right after right only

        # One segment per example
        example_tensor = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "action_labels": action_labels,
            "mention_indice": mention_indice,
            "cluster_category": cluster_category,
            "lr_pair_flag": lr_pair_flag,
            "rr_pair_flag": rr_pair_flag,
            "is_training": is_training
        }
        return doc_key, example_tensor


def coref_collate_fn(batch):
    """
        Collate function for the ERE dataloader.
    """
    doc_keys, batch = zip(*batch)
    batch = {k: [example[k] for example in batch] for k in batch[0]}
    batch_size = len(batch["input_ids"])

    max_input_len = max([example.size(0) for example in batch["input_ids"]])
    max_target_len = max([example.size(0) for example in batch["target_ids"]])

    for k in ["input_ids", "input_mask"]:
        # (batch_size, max_target_len)
        batch[k] = torch.stack([
            F.pad(x, (0, max_input_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)
    for k in ["target_ids", "target_mask", "action_labels",
              "mention_indice", "cluster_category"]:
        if k not in batch:
            continue
        # (batch_size, max_target_len)
        batch[k] = torch.stack([
            F.pad(x, (0, max_target_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)

    max_num_l = max([example.size(1) for example in batch["lr_pair_flag"]])
    max_num_r = max([example.size(1) for example in batch["rr_pair_flag"]])
    
    for k in ["lr_pair_flag"]:
        if k not in batch:
            continue
        # (batch_size, max_target_len, max_num_l)
        if max_num_l > 0:
            batch[k] = torch.stack([
                F.pad(x, (0, max_num_l - x.size(1), 0, max_target_len - x.size(0)), value=0) for x in batch[k]
            ], dim=0)
        else:
            batch[k] = torch.zeros((batch_size, max_target_len, 0), dtype=torch.long)

    for k in ["rr_pair_flag"]:
        if k not in batch:
            continue
        # (batch_size, max_target_len, max_num_r)
        if max_num_r > 0:
            batch[k] = torch.stack([
                F.pad(x, (0, max_num_r - x.size(1), 0, max_target_len - x.size(0)), value=0) for x in batch[k]
            ], dim=0)
        else:
            batch[k] = torch.zeros((batch_size, max_target_len, 0), dtype=torch.long)

    batch["is_training"] = torch.tensor(batch["is_training"], dtype=torch.bool)
    return doc_keys, batch


class CorefDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
