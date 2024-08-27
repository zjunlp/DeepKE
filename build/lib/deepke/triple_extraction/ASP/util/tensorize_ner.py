import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import numpy as np
import random
from transformers import T5Tokenizer
from os.path import join
import json
import copy
import pickle
import logging
import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import util

logger = logging.getLogger(__name__)


class NERDataProcessor(object):
    def __init__(self, config):
        self.config = config

        self.data_dir = config['data_dir']
        self.dataset = config['dataset']

        # Get tensorized samples
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info(
                    f'Loaded tensorized examples from cache: {cache_path}')
        else:
            # Generate tensorized samples
            self.tensor_samples = {}
            tensorizer = Tensorizer(self.config)
            suffix = f'{self.config["plm_tokenizer_name"]}.jsonlines'

            if self.dataset == "conll03_ner":
                paths = {
                    'train': join(self.data_dir, f'train.{suffix}'),
                    'dev': join(self.data_dir, f'dev.{suffix}'),
                    'test': join(self.data_dir, f'test.{suffix}')
                }

            for split, path in paths.items():
                logger.info(
                    f'Tensorizing examples from {path}; results will be cached in {cache_path}')
                is_training = (split == 'train')

                samples = json.load(open(path))
                tensor_samples = [tensorizer.tensorize_example(
                    sample, is_training) for sample in samples]

                self.tensor_samples[split] = NERDataset(
                    sorted(
                        [(doc_key, tensor) for doc_key, tensor in tensor_samples],
                        key=lambda x: -x[1]['input_ids'].size(0)
                    )
                )
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            pickle.dump((self.tensor_samples, self.stored_info),
                        open(cache_path, 'wb'))

    def get_tensor_examples(self):
        # For each split, return list of tensorized samples to allow variable length input (batch size = 1)
        return self.tensor_samples['train'], self.tensor_samples['dev'], self.tensor_samples['test']

    def get_stored_info(self):
        return self.stored_info

    def get_cache_path(self):
        cache_path = join(
            self.data_dir, f'cached.tensors.{self.config["plm_tokenizer_name"]}.bin'
        )
        return cache_path


class Tensorizer:
    def __init__(self, config):
        self.config = config
        self.tz = T5Tokenizer.from_pretrained(config['plm_tokenizer_name'])

        self.num_typing_classes = config['num_typing_classes']

        MENTION_START = '<m>'
        MENTION_END = '</m>'

        self.tz.add_tokens(MENTION_START)
        self.tz.add_tokens(MENTION_END)

        self.mention_start_id = self.tz.convert_tokens_to_ids(MENTION_START)
        self.mention_end_id = self.tz.convert_tokens_to_ids(MENTION_END)

        # Will be used in evaluation
        self.stored_info = {
            'example': {},  # {doc_key: ...}
            'subtoken_maps': {}  # {doc_key: ...}
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
        doc_key = example['doc_id']
        self.stored_info['subtoken_maps'][doc_key] = example.get(
            'subtoken_map', None)
        self.stored_info['example'][doc_key] = example

        is_training = torch.tensor(is_training, dtype=torch.bool)

        # Sentences/segments
        sentence = copy.deepcopy(example['sentence'])  # Segments
        input_sentence = copy.deepcopy(example['input_sentence'])  # Segments
        target_sentence = copy.deepcopy(example["target_sentence"])

        ent_type_sequence = copy.deepcopy(example['ent_type_sequence'])
        ent_indices = copy.deepcopy(example['ent_indices'])

        input_ids = self.tz.convert_tokens_to_ids(input_sentence)
        to_copy_ids = self.tz.convert_tokens_to_ids(sentence)
        target_ids = self.tz.convert_tokens_to_ids(target_sentence)

        input_len, target_len = len(input_ids), len(target_ids)

        input_mask = [1] * input_len
        target_mask = [1] * target_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        to_copy_ids = torch.tensor(to_copy_ids, dtype=torch.long)
        

        target_ids = torch.tensor(target_ids, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.long)

        action_labels = self.get_action_labels(target_ids)

        ent_types = torch.tensor(ent_type_sequence, dtype=torch.long)
        ent_indices = torch.tensor(ent_indices, dtype=torch.long)

        is_l = (target_ids == self.mention_start_id)
        l_ent_indices = ent_indices[is_l]

        # (target_len, num_l)
        lr_pair_flag = (l_ent_indices.unsqueeze(0) == ent_indices.unsqueeze(1))
        # (target_len, num_l)
        # (target_len, 1, num_class) == (target_len, num_l, 1) -> (target_len, num_l, num_class)
        lr_pair_flag = util.one_hot_ignore_negative(
            ent_types, num_classes=self.num_typing_classes
        ).unsqueeze(1) & lr_pair_flag.unsqueeze(-1)

        # Construct example
        example_tensor = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "to_copy_ids": to_copy_ids,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "action_labels": action_labels,
            "ent_indices": ent_indices,
            "ent_types": ent_types,
            "lr_pair_flag": lr_pair_flag,
            "is_training": is_training,
        }

        return doc_key, example_tensor


def ner_collate_fn(batch):
    """
        Collate function for the NER dataloader.
    """
    doc_keys, batch = zip(*batch)
    batch = {k: [example[k] for example in batch] for k in batch[0]}
    batch_size = len(batch["input_ids"])

    max_input_len = max([example.size(0) for example in batch["input_ids"]])
    max_target_len = max([example.size(0) for example in batch["target_ids"]])
    max_sent_id_len = max([example.size(0) for example in batch["to_copy_ids"]])

    for k in ["to_copy_ids"]:
        batch[k] = torch.stack([
            F.pad(x, (0, max_sent_id_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)
    for k in ["input_ids", "input_mask", "sentence_idx"]:
        if k not in batch:
            continue
        # (batch_size, max_target_len)
        batch[k] = torch.stack([
            F.pad(x, (0, max_input_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)
    for k in ["target_ids", "target_mask",
              "ent_indices", "ent_types",
              "action_labels", "target_sentence_idx"]:
        # (batch_size, max_target_len)
        if k not in batch:
            continue
        batch[k] = torch.stack([
            F.pad(x, (0, max_target_len - x.size(0)), value=0) for x in batch[k]
        ], dim=0)

    max_num_l = max([example.size(1) for example in batch["lr_pair_flag"]])

    for k in ["lr_pair_flag"]:
        # (batch_size, max_target_len, max_num_l, num_class)
        if max_num_l > 0:
            batch[k] = torch.stack([
                F.pad(x, (0, 0, 0, max_num_l - x.size(1), 0, max_target_len - x.size(0)), value=0) for x in batch[k]
            ], dim=0)
        else:
            batch[k] = torch.zeros(
                (batch_size, max_target_len, 0), dtype=torch.long)

    batch["is_training"] = torch.tensor(batch["is_training"], dtype=torch.bool)
    return doc_keys, batch


class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
