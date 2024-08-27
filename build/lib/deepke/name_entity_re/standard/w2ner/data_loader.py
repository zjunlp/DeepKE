import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from tqdm import tqdm
from transformers import AutoTokenizer
import os
from deepke.name_entity_re.standard.tools import *
import requests
from typing import List
from .utils import convert_index_to_text
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab, mode):

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in tqdm(enumerate(data), desc=f'preprocess {mode} data'):
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        _entity_text = set([convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def trans_BIO_Dataset(config, examples: List[InputExample], mode):
    D = []
    for example in examples:
        span_infos = []
        sentence, label, d = example.text_a.split(' '), example.label, []
        if len(sentence) > config.max_seq_len:
            continue
        assert len(sentence) == len(label)
        i = 0
        while i < len(sentence):
            flag = label[i]
            if flag[0] == 'B':
                start_index = i
                i+=1
                while(i < len(sentence) and label[i][0] == 'I'):
                    i+=1
                d.append([start_index, i, flag[2:]])
            elif flag[0] == 'I':
                start_index = i
                i+=1
                while(i < len(sentence) and label[i][0] == 'I'):
                    i+=1
                d.append([start_index, i, flag[2:]])
            else:
                i+=1
        for s_e_flag in d:
            start_span, end_span, flag = s_e_flag[0], s_e_flag[1], s_e_flag[2]
            span_infos.append({'index': list(range(start_span, end_span)), 'type': flag})
        D.append({'sentence': sentence, 'ner': span_infos})

    config.logger.info(f'{mode} dataset example')
    for example in D[:2]:
        config.logger.info("\n{}".format(example))
    return D




def load_data_bert(config, train_examples: List[InputExample] = None, eval_examples: List[InputExample] = None,
                   test_examples: List[InputExample] = None):
    train_data = trans_BIO_Dataset(config, train_examples, 'train') if train_examples else None
    dev_data = trans_BIO_Dataset(config, eval_examples, 'dev') if eval_examples else None
    test_data = trans_BIO_Dataset(config, test_examples, 'test') if test_examples else None

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data) if train_data else None
    dev_ent_num = fill_vocab(vocab, dev_data) if dev_data else None
    test_ent_num = fill_vocab(vocab, test_data) if test_data else None

    table = pt.PrettyTable(['Dataset', 'sentences', 'entities'])
    if train_data:
        table.add_row(['train', len(train_data), train_ent_num])
    if dev_data:
        table.add_row(['dev', len(dev_data), dev_ent_num])
    if test_data:
        table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    print(vocab.label2id)
    config.vocab = vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab, 'train')) if train_data else None
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab, 'dev')) if dev_data else None
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab, 'test')) if test_data else None
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)
