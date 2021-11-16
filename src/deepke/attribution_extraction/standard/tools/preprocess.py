import os
import logging
from collections import OrderedDict
from typing import List, Dict
from transformers import BertTokenizer
from .serializer import Serializer
from .vocab import Vocab
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import save_pkl, load_csv

logger = logging.getLogger(__name__)

__all__ = [
    "_handle_pos_limit",
    "_add_pos_seq",
    "_convert_tokens_into_index",
    "_serialize_sentence",
    "_lm_serialize",
    "_add_attribute_data",
    "_handle_attribute_data",
    "preprocess"
]
def _handle_pos_limit(pos: List[int], limit: int) -> List[int]:
    for i,p in enumerate(pos):
        if p > limit:
            pos[i] = limit
        if p < -limit:
            pos[i] = -limit
    return [p + limit + 1 for p in pos]

def _add_pos_seq(train_data: List[Dict], cfg):
    for d in train_data:
        entities_idx = [d['entity_index'],d['attribute_value_index']
                ] if d['entity_index'] < d['attribute_value_index'] else [d['entity_index'], d['attribute_value_index']]

        d['entity_pos'] = list(map(lambda i: i - d['entity_index'], list(range(d['seq_len']))))
        d['entity_pos'] = _handle_pos_limit(d['entity_pos'],int(cfg.pos_limit))

        d['attribute_value_pos'] = list(map(lambda i: i - d['attribute_value_index'], list(range(d['seq_len']))))
        d['attribute_value_pos'] = _handle_pos_limit(d['attribute_value_pos'],int(cfg.pos_limit))

        if cfg.model_name == 'cnn':
            if cfg.use_pcnn:
                d['entities_pos'] = [1] * (entities_idx[0] + 1) + [2] * (entities_idx[1] - entities_idx[0] - 1) +\
                                    [3] * (d['seq_len'] - entities_idx[1])

def _convert_tokens_into_index(data: List[Dict], vocab):
    unk_str = '[UNK]'
    unk_idx = vocab.word2idx[unk_str]

    for d in data:
        d['token2idx'] = [vocab.word2idx.get(i, unk_idx) for i in d['tokens']]
        d['seq_len'] = len(d['token2idx'])

def _serialize_sentence(data: List[Dict], serial):
    for d in data:
        sent = d['sentence'].strip()
        sent = sent.replace(d['entity'] , ' entity ' , 1).replace(d['attribute_value'] , ' attribute_value ' , 1)
        d['tokens'] = serial(sent, never_split=['entity','attribute_value'])
        entity_index, attribute_value_index = d['entity_offset'] , d['attribute_value_offset']
        d['entity_index'],d['attribute_value_index'] = int(entity_index) , int(attribute_value_index)
       
def _lm_serialize(data: List[Dict], cfg):
    logger.info('use bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm_file)
    for d in data:
        sent = d['sentence'].strip()
        sent += '[SEP]' + d['entity'] + '[SEP]' + d['attribute_value']
        d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)
        d['seq_len'] = len(d['token2idx'])

def _add_attribute_data(atts: Dict, data: List) -> None:
    for d in data:
        d['att2idx'] = atts[d['attribute']]['index']

def _handle_attribute_data(attribute_data: List[Dict]) -> Dict:
    atts = OrderedDict()
    attribute_data = sorted(attribute_data, key=lambda i: int(i['index']))
    for d in attribute_data:
        atts[d['attribute']] = {
            'index': int(d['index'])
        }
    return atts

def preprocess(cfg):
    logger.info('===== start preprocess data =====')
    train_fp = os.path.join(cfg.cwd, cfg.data_path, 'train.csv')
    valid_fp = os.path.join(cfg.cwd, cfg.data_path, 'valid.csv')
    test_fp = os.path.join(cfg.cwd, cfg.data_path, 'test.csv')
    attribute_fp = os.path.join(cfg.cwd, cfg.data_path, 'attribute.csv')

    logger.info('load raw files...')
    train_data = load_csv(train_fp)
    valid_data = load_csv(valid_fp)
    test_data = load_csv(test_fp)
    attribute_data = load_csv(attribute_fp)

    logger.info('convert attribution into index...')
    atts = _handle_attribute_data(attribute_data)
    _add_attribute_data(atts,train_data)
    _add_attribute_data(atts,test_data)
    _add_attribute_data(atts,valid_data)

    logger.info('verify whether use pretrained language models...')
    if cfg.model_name == 'lm':
        logger.info('use pretrained language models serialize sentence...')
        _lm_serialize(train_data, cfg)
        _lm_serialize(valid_data, cfg)
        _lm_serialize(test_data, cfg)
    else:
        logger.info('serialize sentence into tokens...')
        serializer = Serializer(do_chinese_split=cfg.chinese_split, do_lower_case=True)
        serial = serializer.serialize
        _serialize_sentence(train_data, serial)
        _serialize_sentence(valid_data, serial)
        _serialize_sentence(test_data, serial)

        logger.info('build vocabulary...')
        vocab = Vocab('word')
        train_tokens = [d['tokens'] for d in train_data]
        valid_tokens = [d['tokens'] for d in valid_data]
        test_tokens = [d['tokens'] for d in test_data]
        sent_tokens = [*train_tokens, *valid_tokens, *test_tokens]

        for sent in sent_tokens:
            vocab.add_words(sent)
        vocab.trim(min_freq=cfg.min_freq)

        _convert_tokens_into_index(train_data, vocab)
        _convert_tokens_into_index(valid_data, vocab)
        _convert_tokens_into_index(test_data, vocab)

        logger.info('build position sequence...')
        _add_pos_seq(train_data, cfg)
        _add_pos_seq(valid_data, cfg)
        _add_pos_seq(test_data, cfg)

    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, cfg.out_path), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)

    if cfg.model_name != 'lm':
        vocab_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl')
        vocab_txt = os.path.join(cfg.cwd, cfg.out_path, 'vocab.txt')
        save_pkl(vocab, vocab_save_fp)
        logger.info('save vocab in txt file, for watching...')
        with open(vocab_txt, 'w', encoding='utf-8') as f:
            f.write(os.linesep.join(vocab.word2idx.keys()))

    logger.info('===== end preprocess data =====')


