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

class EREDataProcessor(object):
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
                logger.info(f'Loaded tensorized examples from cache: {cache_path}')
        else:
            # Generate tensorized samples
            self.tensor_samples = {}
            tensorizer = Tensorizer(self.config)
            suffix = f'{self.config["plm_tokenizer_name"]}.jsonlines'

            if self.dataset == "conll04":
                paths = {
                    'trn': join(self.data_dir, f'train_dev.{suffix}'),
                    'dev': join(self.data_dir, f'dev.{suffix}'),
                    'tst': join(self.data_dir, f'test.{suffix}')
                }
            elif self.dataset == "CMeIE":
                paths = {
                    'trn': join(self.data_dir, f'train.{suffix}'),
                    'dev': join(self.data_dir, f'dev.{suffix}'),
                    'tst': join(self.data_dir, f'test.{suffix}')
                }

            for split, path in paths.items():
                logger.info(f'Tensorizing examples from {path}; results will be cached in {cache_path}')
                is_training = (split == 'trn')

                with open(path, 'r') as f:
                    samples = json.loads(f.read())
                tensor_samples = [tensorizer.tensorize_example(sample, is_training) for sample in samples]
                print(split, len(tensor_samples))

                self.tensor_samples[split] = EREDataset(
                    sorted(
                        [(doc_key, tensor) for doc_key, tensor in tensor_samples],
                        key=lambda x: -x[1]['input_ids'].size(0)
                    )
                )
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)

    def get_tensor_examples(self):
        # For each split, return list of tensorized samples to allow variable length input (batch size = 1)
        return self.tensor_samples['trn'], self.tensor_samples['dev'], self.tensor_samples['tst']

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
        self.num_linking_classes = config['num_linking_classes']

        MENTION_START = '<m>'
        MENTION_END   = '</m>'

        self.tz.add_tokens(MENTION_START)
        self.tz.add_tokens(MENTION_END)

        self.mention_start_id = self.tz.convert_tokens_to_ids(MENTION_START)
        self.mention_end_id   = self.tz.convert_tokens_to_ids(MENTION_END)

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
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        self.stored_info['example'][doc_key] = example
        
        is_training = torch.tensor(is_training, dtype=torch.bool)

        # Sentences/segments
        sentence = copy.deepcopy(example['sentence'])  # Segments
        input_sentence = copy.deepcopy(example['input_sentence'])  # Segments
        target_sentence = copy.deepcopy(example["target_sentence"])
        
        ent_type_sequence = copy.deepcopy(example['ent_type_sequence'])
        ent_indices = copy.deepcopy(example['ent_indices'])
        rel_type_sequence = copy.deepcopy(example['rel_type_sequence'])
        rel_indices = copy.deepcopy(example['rel_indices'])
        
        input_ids = self.tz.convert_tokens_to_ids(input_sentence)
        to_copy_ids = self.tz.convert_tokens_to_ids(sentence)
        target_ids = self.tz.convert_tokens_to_ids(target_sentence)

        input_len, target_len = len(input_ids), len(target_ids)

        input_mask = [1] * input_len
        target_mask = [1] * target_len
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        to_copy_ids = torch.tensor(to_copy_ids, dtype=torch.long)
        
        target_ids  = torch.tensor(target_ids, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.long)
        if "sentence_idx" in example:
            # for multi-sentence input
            sentence_idx = torch.tensor(example['sentence_idx'], dtype=torch.long)
            target_sentence_idx = torch.tensor(example['target_sentence_idx'], dtype=torch.long)

        action_labels = self.get_action_labels(target_ids)
        
        ent_types = torch.tensor(ent_type_sequence, dtype=torch.long)
        ent_indices = torch.tensor(ent_indices, dtype=torch.long)
        
        rel_types = torch.tensor(rel_type_sequence, dtype=torch.long)
        rel_indices = torch.tensor(rel_indices, dtype=torch.long)

        linearized_indices = torch.arange(target_len)

        is_l = (target_ids == self.mention_start_id)
        is_r = (target_ids == self.mention_end_id)

        l_ent_indices = ent_indices[is_l]

        # (target_len, num_l)
        lr_pair_flag = (l_ent_indices.unsqueeze(0) == ent_indices.unsqueeze(1))
        # (target_len, num_l)
        # (target_len, 1, num_class) == (target_len, num_l, 1) -> (target_len, num_l, num_class)
        lr_pair_flag = util.one_hot_ignore_negative(
            ent_types, num_classes=self.num_typing_classes
        ).unsqueeze(1) & lr_pair_flag.unsqueeze(-1)

        # (num_r)
        r_pos = linearized_indices[is_r]
        # (target_len, num_r)
        distance_to_previous_r = linearized_indices.unsqueeze(1) - r_pos.unsqueeze(0)
        # (target_len, num_r)
        is_after_r = (distance_to_previous_r > 0)

        # (target_len, 1, num_rel) == (1, num_r, 1) -> (target_len, num_r, num_rel)
        rel_same = (
            rel_indices.unsqueeze(1) == r_pos.unsqueeze(0).unsqueeze(-1)
        )

        # (target_len, num_rel, num_linking_classes)
        
        
        oh_rel_types = util.one_hot_ignore_negative(
            rel_types, num_classes=2*self.num_linking_classes
        )
        # (target_len, 1, num_rel, num_linking_classes) & (target_len, num_r, num_rel, 1)
        # -> (target_len, num_r, num_rel, num_linking_classes)
        oh_rel_types = oh_rel_types.unsqueeze(1) & rel_same.unsqueeze(-1)
        # (target_len, num_r, num_linking_classes)
        rr_pair_flag = torch.any(oh_rel_types, dim=2) & is_after_r.unsqueeze(-1)

        # (target_len, num_r, num_linking_classes+1)
        rr_pair_flag = torch.cat([
            ~torch.any(rr_pair_flag, dim=2, keepdim=True),
            rr_pair_flag], dim=2
        )

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
            "rel_indices": rel_indices,
            "rel_types": rel_types,
            "lr_pair_flag": lr_pair_flag,
            "rr_pair_flag": rr_pair_flag,
            "is_training": is_training,
        }
        if "sentence_idx" in example:
            # for multi-sentence input
            example_tensor["sentence_idx"] = sentence_idx
            example_tensor["target_sentence_idx"] = target_sentence_idx
            # (target_len, 1) == (1, num_r) -> (target_len, num_r)
            same_sentence_flag = (target_sentence_idx.unsqueeze(1) == target_sentence_idx[r_pos].unsqueeze(0))
            example_tensor["same_sentence_flag"] = same_sentence_flag

        return doc_key, example_tensor


def ere_collate_fn(batch):
    """
        Collate function for the ERE dataloader.
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
    max_num_r = max([example.size(1) for example in batch["rr_pair_flag"]])
    max_num_relation = max([example.size(1) for example in batch["rel_indices"]])

    for k in ["lr_pair_flag"]:
        # (batch_size, max_target_len, max_num_l, num_class)
        if max_num_l > 0:
            batch[k] = torch.stack([
                F.pad(x, (0,0, 0, max_num_l - x.size(1),0, max_target_len - x.size(0)), value=0) for x in batch[k]
            ], dim=0)
        else:
            batch[k] = torch.zeros(
                (batch_size, max_target_len, 0), dtype=torch.long)

    for k in ["rr_pair_flag"]:
        # (batch_size, max_target_len, max_num_r, 1+num_linking_classes)
        if max_num_r > 0:
            # (batch_size, max_target_len, max_num_r, 1+num_linking_classes)
            batch[k] = torch.stack([
                F.pad(x, (0, 0, 0, max_num_r - x.size(1), 0, max_target_len - x.size(0)), value=0) for x in batch[k]
            ], dim=0)
            batch[k][:,:,:,0] |= ~torch.any(batch[k][:,:,:,1:], dim=3)
        else:
            batch[k] = torch.zeros(
                (batch_size, max_target_len, 0), dtype=torch.long)

    for k in ["same_sentence_flag"]:
        if k not in batch:
            continue
        # (batch_size, max_target_len, max_num_r)
        if max_num_r > 0:
            batch[k] = torch.stack([
                F.pad(x, (0, max_num_r - x.size(1), 0, max_target_len - x.size(0)), value=0) for x in batch[k]
            ], dim=0)
        else:
            batch[k] = torch.zeros(
                (batch_size, max_target_len, 0), dtype=torch.long)

    for k in ["rel_indices", "rel_types"]:
        if max_num_relation > 0:
            batch[k] = torch.stack([
                F.pad(x, (0, max_num_relation - x.size(1), 0, max_target_len - x.size(0)), value=-1) for x in batch[k]
            ], dim=0)
        else:
            batch[k] = torch.zeros(
                (batch_size, max_target_len, 0), dtype=torch.bool)

    batch["is_training"] = torch.tensor(batch["is_training"], dtype=torch.bool)
    return doc_keys, batch


class EREDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

