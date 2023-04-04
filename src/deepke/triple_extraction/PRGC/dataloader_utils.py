# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain
import os

from .util import Label2IdxSub, Label2IdxObj


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag


def read_examples(data_dir, data_sign, rel2idx):
    """load data to InputExamples
    """
    examples = []

    # read src data
    with open(os.path.join(data_dir, f'{data_sign}_triples.json'), "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            rel2ens = defaultdict(list)
            en_pair_list = []
            re_list = []

            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]])
                re_list.append(rel2idx[triple[1]])
                rel2ens[rel2idx[triple[1]]].append((triple[0], triple[-1]))
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def _get_so_head(en_pair, tokenizer, text_tokens):
    sub = tokenizer.tokenize(en_pair[0])
    obj = tokenizer.tokenize(en_pair[1])
    sub_head = find_head_idx(source=text_tokens, target=sub)
    if sub == obj:
        obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = sub_head
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj)
    return sub_head, obj_head, sub, obj


def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params):
    """convert function
    """
    text_tokens = tokenizer.tokenize(example.text)
    # cut off
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation
        corres_tag = np.zeros((max_text_len, max_text_len))
        rel_tag = len(rel2idx) * [0]
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            # get sub and obj head
            sub_head, obj_head, _, _ = _get_so_head(en_pair, tokenizer, text_tokens)
            # construct relation tag
            rel_tag[rel] = 1
            if sub_head != -1 and obj_head != -1:
                corres_tag[sub_head][obj_head] = 1

        sub_feats = []
        # positive samples
        for rel, en_ll in example.rel2ens.items():
            # init
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxSub['O']]
            for en in en_ll:
                # get sub and obj head
                sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            seq_tag = [tags_sub, tags_obj]

            # sanity check
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'
            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corres_tag=corres_tag,
                seq_tag=seq_tag,
                relation=rel,
                rel_tag=rel_tag
            ))
        # relation judgement ablation
        if not ex_params['ensure_rel']:
            # negative samples
            neg_rels = set(rel2idx.values()).difference(set(example.re_list))
            neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
            for neg_rel in neg_rels:
                # init
                seq_tag = max_text_len * [Label2IdxSub['O']]
                # sanity check
                assert len(input_ids) == len(seq_tag) == len(attention_mask) == max_text_len, f'length is not equal!!'
                seq_tag = [seq_tag, seq_tag]
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=neg_rel,
                    rel_tag=rel_tag
                ))
    # val and test data
    else:
        triples = []
        for rel, en in zip(example.re_list, example.en_pair_list):
            # get sub and obj head
            sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats


def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = params.max_seq_length
    # multi-process
    with Pool(10) as p:
        convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                                         data_sign=data_sign, ex_params=ex_params)
        features = p.map(func=convert_func, iterable=examples)

    return list(chain(*features))
