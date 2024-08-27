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
    "_add_relation_data",
    "_handle_relation_data",
    "preprocess"
]
def _handle_pos_limit(pos: List[int], limit: int) -> List[int]:
    """
    处理句子长度，设定句长限制
    Args :
        pos (List[int]) : 句子对应的List
        limit (int) : 限制的数
    Return :
        [p + limit + 1 for p in pos] (List[int]) : 处理后的结果

    """
    for i, p in enumerate(pos):
        if p > limit:
            pos[i] = limit
        if p < -limit:
            pos[i] = -limit
    return [p + limit + 1 for p in pos]


def _add_pos_seq(train_data: List[Dict], cfg):
    """
    增加位置序列
    Args : 
        train_data (List[Dict]) : 数据集合
        cfg : 配置文件
    """
    for d in train_data:
        entities_idx = [d['head_idx'], d['tail_idx']
                        ] if d['head_idx'] < d['tail_idx'] else [d['tail_idx'], d['head_idx']]

        d['head_pos'] = list(map(lambda i: i - d['head_idx'], list(range(d['seq_len']))))
        d['head_pos'] = _handle_pos_limit(d['head_pos'], int(cfg.pos_limit))

        d['tail_pos'] = list(map(lambda i: i - d['tail_idx'], list(range(d['seq_len']))))
        d['tail_pos'] = _handle_pos_limit(d['tail_pos'], int(cfg.pos_limit))

        if cfg.model_name == 'cnn':
            if cfg.use_pcnn:
                # 当句子无法分隔成三段时，无法使用PCNN
                # 比如： [head, ... tail] or [... head, tail, ...] 无法使用统一方式 mask 分段
                d['entities_pos'] = [1] * (entities_idx[0] + 1) + [2] * (entities_idx[1] - entities_idx[0] - 1) +\
                                    [3] * (d['seq_len'] - entities_idx[1])


def _convert_tokens_into_index(data: List[Dict], vocab):
    """
    将tokens转换成index值
    Args : 
        data (List[Dict]) : 数据集合
        vocab (Class) : 词汇表
    """
    unk_str = '[UNK]'
    unk_idx = vocab.word2idx[unk_str]

    for d in data:
        d['token2idx'] = [vocab.word2idx.get(i, unk_idx) for i in d['tokens']]
        d['seq_len'] = len(d['token2idx'])


def _serialize_sentence(data: List[Dict], serial, cfg):
    """
    将句子分词
    Args : 
        data (List[Dict]) : 数据集合
        serial (Class): Serializer类
        cfg : 配置文件
    """
    for d in data:
        sent = d['sentence'].strip()
        if d['head'] in d['tail']:
            sent = sent.replace(d['tail'], ' tail ', 1).replace(d['head'], ' head ', 1)
        else:
            sent = sent.replace(d['head'], ' head ', 1).replace(d['tail'], ' tail ', 1)
        d['tokens'] = serial(sent, never_split=['head', 'tail'])
        head_idx, tail_idx = d['tokens'].index('head'), d['tokens'].index('tail')
        d['head_idx'], d['tail_idx'] = head_idx, tail_idx

        if cfg.replace_entity_with_type:
            if cfg.replace_entity_with_scope:
                d['tokens'][head_idx], d['tokens'][tail_idx] = 'HEAD_' + d['head_type'], 'TAIL_' + d['tail_type']
            else:
                d['tokens'][head_idx], d['tokens'][tail_idx] = d['head_type'], d['tail_type']
        else:
            if cfg.replace_entity_with_scope:
                d['tokens'][head_idx], d['tokens'][tail_idx] = 'HEAD', 'TAIL'
            else:
                d['tokens'][head_idx], d['tokens'][tail_idx] = d['head'], d['tail']


def _lm_serialize(data: List[Dict], cfg):
    """
    lm模型分词
    Args : 
        data (List[Dict]) : 数据集合
        cfg : 配置文件
    """
    logger.info('use bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm_file)
    for d in data:
        sent = d['sentence'].strip()
        sent = sent.replace(d['head'], d['head_type'], 1).replace(d['tail'], d['tail_type'], 1)
        sent += '[SEP]' + d['head'] + '[SEP]' + d['tail']
        d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)
        d['seq_len'] = len(d['token2idx'])


def _add_relation_data(rels: Dict, data: List) -> None:
    """
    增加关系数据
    Args :
        rels (Dict) : 关系字典集合
        data (List) : 所需增加的关系数据
    """
    for d in data:
        d['rel2idx'] = rels[d['relation']]['index']
        d['head_type'] = rels[d['relation']]['head_type']
        d['tail_type'] = rels[d['relation']]['tail_type']


def _handle_relation_data(relation_data: List[Dict]) -> Dict:
    """
    处理关系数据，每一个关系有index，head_type,tail_type三个属性
    Args: 
        relation_data (List[Dict]) : 所需要处理的关系数据
    Return :
        rels (Dict) : 处理之后的结果
    """
    rels = OrderedDict()
    relation_data = sorted(relation_data, key=lambda i: int(i['index']))
    for d in relation_data:
        rels[d['relation']] = {
            'index': int(d['index']),
            'head_type': d['head_type'],
            'tail_type': d['tail_type'],
        }

    return rels

def _clean_data(data):
    """
    清洗数据，去除一些头尾实体不存在的句子
    Args: 
        data((List[Dict])): 需要处理的数据
    Returns:
        clean_data : 处理后的数据
    """
    clean_data = []
    for d in data:
        sent = d['sentence']
        head = d['head']
        tail = d['tail']
        if head in sent and tail in sent:
            clean_data.append(d)
    return clean_data

def preprocess(cfg):
    """
    数据预处理阶段
    """
    logger.info('===== start preprocess data =====')
    train_fp = os.path.join(cfg.cwd, cfg.data_path, 'train.csv')
    valid_fp = os.path.join(cfg.cwd, cfg.data_path, 'valid.csv')
    test_fp = os.path.join(cfg.cwd, cfg.data_path, 'test.csv')
    relation_fp = os.path.join(cfg.cwd, cfg.data_path, 'relation.csv')

    logger.info('load raw files...')
    train_data = load_csv(train_fp)
    valid_data = load_csv(valid_fp)
    test_data = load_csv(test_fp)
    relation_data = load_csv(relation_fp)

    logger.info('clean data...')
    train_data = _clean_data(train_data)
    valid_data = _clean_data(valid_data)
    test_data = _clean_data(test_data)

    logger.info('convert relation into index...')
    rels = _handle_relation_data(relation_data)
    _add_relation_data(rels, train_data)
    _add_relation_data(rels, valid_data)
    _add_relation_data(rels, test_data)

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
        _serialize_sentence(train_data, serial, cfg)
        _serialize_sentence(valid_data, serial, cfg)
        _serialize_sentence(test_data, serial, cfg)

        logger.info('build vocabulary...')
        vocab = Vocab('word')
        train_tokens = [d['tokens'] for d in train_data]
        valid_tokens = [d['tokens'] for d in valid_data]
        test_tokens = [d['tokens'] for d in test_data]
        sent_tokens = [*train_tokens, *valid_tokens, *test_tokens]
        for sent in sent_tokens:
            vocab.add_words(sent)
        vocab.trim(min_freq=cfg.min_freq)

        logger.info('convert tokens into index...')
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



