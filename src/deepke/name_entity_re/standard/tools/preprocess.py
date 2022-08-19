from .dataset import *

import argparse
import csv
import json
import logging
import os
import random
import sys
import numpy as np
from hydra import utils
import pickle

class NerProcessor(DataProcessor):
    """Processor for the dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, cfg):
        labels = ['O']
        for i in cfg.labels:
            labels.append('B-'+i)
            labels.append('I-'+i)
        labels.append('[CLS]')
        labels.append('[SEP]')
        return labels

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features

def build_crflstm_corpus(split, cfg):
    assert split in ['train', 'dev', 'test']
    processor = NerProcessor()
    if split == 'train':
        examples = processor.get_train_examples(os.path.join(utils.get_original_cwd(), cfg.data_dir))
        word2id = {}
        label2id = {}
        id2label = {}
        for example in examples:
            textlist = example.text_a.split(' ')
            labellist = example.label
            for text in textlist:
                if text not in word2id:
                    word2id[text] = len(word2id)
            for label in labellist:
                if label not in label2id:
                    label2id[label] = len(label2id)
                    id2label[len(label2id) - 1] = label
        word2id['<unk>'] = len(word2id)
        word2id['<pad>'] = len(word2id)
        with open(os.path.join(utils.get_original_cwd(), cfg.data_dir, cfg.model_vocab_path), 'wb') as outp:
            pickle.dump(word2id, outp)
            pickle.dump(label2id, outp)
            pickle.dump(id2label, outp)
        return examples, word2id, label2id, id2label

    elif split == 'dev':
        examples = processor.get_dev_examples(os.path.join(utils.get_original_cwd(), cfg.data_dir))
        return examples
    else:
        examples = processor.get_test_examples(os.path.join(utils.get_original_cwd(), cfg.data_dir))
        return examples
