import os
import csv
import json
import torch
import jieba
import logging
from typing import List, Tuple
# self file
from deepke.config import config
from deepke.vocab import Vocab
from deepke.utils import ensure_dir, save_pkl, load_csv, load_jsonld
from pytorch_transformers import BertTokenizer

jieba.setLogLevel(logging.INFO)


def build_lm_data(raw_data: List) -> List:
    tokenizer = BertTokenizer.from_pretrained(config.lm_name)
    sents = []
    for data in raw_data:
        sent = data[0]
        sub = data[1]
        obj = data[4]
        sent = '[CLS]' + sent + '[SEP]' + sub + '[SEP]' + obj + '[SEP]'
        input_ids = torch.tensor([tokenizer.encode(sent)])
        sents.append(input_ids)
    return sents


def mask_feature(entities_pos: List, sen_len: int) -> List:
    left = [1] * (entities_pos[0] + 1)
    middle = [2] * (entities_pos[1] - entities_pos[0] - 1)
    right = [3] * (sen_len - entities_pos[1])
    return left + middle + right


def pos_feature(sent_len: int, entity_pos: int, entity_len: int,
                pos_limit: int) -> List:
    left = list(range(-entity_pos, 0))
    middle = [0] * entity_len
    right = list(range(1, sent_len - entity_pos - entity_len + 1))
    pos = left + middle + right
    for i, p in enumerate(pos):
        if p > pos_limit:
            pos[i] = pos_limit
        if p < -pos_limit:
            pos[i] = -pos_limit
    pos = [p + pos_limit + 1 for p in pos]
    return pos


def build_data(raw_data: List[List], vocab) -> Tuple[List, List, List, List]:
    sents = []
    head_pos = []
    tail_pos = []
    mask_pos = []

    if vocab.name == 'word':
        for data in raw_data:
            sent = [vocab.word2idx.get(w, 1) for w in data[-2]]
            pos = list(range(len(sent)))
            head, tail = int(data[-1][0]), int(data[-1][1])
            entities_pos = [head, tail] if tail > head else [tail, head]
            head_p = pos_feature(len(sent), head, 1, config.pos_limit)
            tail_p = pos_feature(len(sent), tail, 1, config.pos_limit)
            mask_p = mask_feature(entities_pos, len(sent))
            sents.append(sent)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)

    else:
        for data in raw_data:
            sent = [vocab.word2idx.get(w, 1) for w in data[0]]
            head, tail = int(data[3]), int(data[6])
            head_len, tail_len = len(data[1]), len(data[4])
            entities_pos = [head, tail] if tail > head else [tail, head]
            head_p = pos_feature(len(sent), head, head_len, config.pos_limit)
            tail_p = pos_feature(len(sent), tail, tail_len, config.pos_limit)
            mask_p = mask_feature(entities_pos, len(sent))
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)
            sents.append(sent)
    return sents, head_pos, tail_pos, mask_pos


def relation_tokenize(relations: List[str], fp: str) -> List[int]:
    rels_arr = []
    rels = {}
    out = []
    with open(fp, encoding='utf-8') as f:
        for l in f:
            rels_arr.append(l.strip())
    for i, rel in enumerate(rels_arr):
        rels[rel] = i
    for rel in relations:
        out.append(rels[rel])
    return out


def build_vocab(raw_data: List[List], out_path: str) -> Tuple[Vocab, str]:
    if config.word_segment:
        vocab = Vocab('word')
        for data in raw_data:
            vocab.add_sent(data[-2])
    else:
        vocab = Vocab('char')
        for data in raw_data:
            vocab.add_sent(data[0])
    vocab.trim(config.min_freq)

    ensure_dir(out_path)
    vocab_path = os.path.join(out_path, 'vocab.pkl')
    vocab_txt = os.path.join(out_path, 'vocab.txt')
    save_pkl(vocab_path, vocab, 'vocab')
    with open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join([word for word in vocab.word2idx.keys()]))
    return vocab, vocab_path


def split_sents(raw_data: List[List], verbose: bool = True) -> List[List]:
    if verbose:
        print('need word segment, use jieba to split sentence')
    new_data = []
    jieba.add_word('HEAD')
    jieba.add_word('TAIL')
    for data in raw_data:
        head, tail = data[2], data[5]
        sent = data[0].replace(data[1], 'HEAD', 1)
        sent = sent.replace(data[4], 'TAIL', 1)
        sent = jieba.lcut(sent)
        head_pos, tail_pos = sent.index('HEAD'), sent.index('TAIL')
        sent[head_pos] = head
        sent[tail_pos] = tail
        data.append(sent)
        data.append([head_pos, tail_pos])
        new_data.append(data)
    return new_data


def exist_relation(fp: str, file_type: str) -> int:
    '''
    判断文件是否存在关系数据，即判断文件是用来训练还是用来预测
    当存在关系数据时，返回对应所在的列值（int number >= 0)
    当不存在时，返回 -1
    :param fp: 文件地址
    :return: 数值
    '''
    with open(fp, encoding='utf-8') as f:
        if file_type == 'csv':
            f = csv.DictReader(f)
        for l in f:
            if file_type == 'jsonld':
                l = json.loads(l)
            keys = list(l.keys())
            try:
                num = keys.index('relation')
            except:
                num = -1
            return num


def process(data_path: str, out_path: str, file_type: str) -> None:
    print('===== start preprocess data =====')

    file_type = file_type.lower()
    assert file_type in ['csv', 'jsonld']

    print('load raw files...')
    train_fp = os.path.join(data_path, 'train.' + file_type)
    test_fp = os.path.join(data_path, 'test.' + file_type)
    relation_fp = os.path.join(data_path, 'relation.txt')

    relation_place = exist_relation(train_fp, file_type)
    if file_type == 'csv':
        train_raw_data = load_csv(train_fp)
        test_raw_data = load_csv(test_fp)
    else:
        train_raw_data = load_jsonld(train_fp)
        test_raw_data = load_jsonld(test_fp)
    train_relation = []
    test_relation = []
    if relation_place > -1:
        for data in train_raw_data:
            train_relation.append(data.pop(relation_place))
        for data in test_raw_data:
            test_relation.append(data.pop(relation_place))

    # 使用语言模型预训练时
    if config.model_name == 'Bert':
        train_lm_sents = build_lm_data(train_raw_data)
        test_lm_sents = build_lm_data(test_raw_data)

    # 当为中文时是否需要分词操作，如果sentence已经为分词的结果，则不需要分词
    print('\nverify whether need split words...')
    if config.is_chinese and config.word_segment:
        train_raw_data = split_sents(train_raw_data)
        test_raw_data = split_sents(test_raw_data, verbose=False)

    print('build sentence vocab...')
    vocab, vocab_path = build_vocab(train_raw_data, out_path)

    print('\nbuild train data...')
    train_sents, train_head_pos, train_tail_pos, train_mask_pos = build_data(
        train_raw_data, vocab)
    print('build test data...')
    test_sents, test_head_pos, test_tail_pos, test_mask_pos = build_data(
        test_raw_data, vocab)
    print('build relation data...\n')
    train_rel_tokens = relation_tokenize(train_relation, relation_fp)
    test_rel_tokens = relation_tokenize(test_relation, relation_fp)

    train_data = list(
        zip(train_sents, train_head_pos, train_tail_pos, train_mask_pos,
            train_rel_tokens))
    test_data = list(
        zip(test_sents, test_head_pos, test_tail_pos, test_mask_pos,
            test_rel_tokens))

    if config.model_name == 'Bert':
        train_data = list(zip(train_lm_sents, train_rel_tokens))
        test_data = list(zip(test_lm_sents, test_rel_tokens))

    ensure_dir(out_path)
    train_data_path = os.path.join(out_path, 'train.pkl')
    test_data_path = os.path.join(out_path, 'test.pkl')

    save_pkl(train_data_path, train_data, 'train data')
    save_pkl(test_data_path, test_data, 'test data')

    if config.model_name == 'Bert':
        train_lm_data_path = os.path.join(out_path, 'train_lm.pkl')
        test_lm_data_path = os.path.join(out_path, 'test_lm.pkl')

        save_pkl(train_lm_data_path, train_data, 'train data')
        save_pkl(test_lm_data_path, test_data, 'test data')

    print('===== end preprocess data =====')


if __name__ == "__main__":
    data_path = '../data/origin'
    out_path = '../data/out'

    process(data_path, out_path, file_type='csv')
