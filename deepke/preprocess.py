import os
import jieba
import logging
from typing import List, Dict
from pytorch_transformers import BertTokenizer
# self file
from deepke.vocab import Vocab
from deepke.config import config
from deepke.utils import ensure_dir, save_pkl, load_csv

jieba.setLogLevel(logging.INFO)

Path = str


def _mask_feature(entities_idx: List, sen_len: int) -> List:
    left = [1] * (entities_idx[0] + 1)
    middle = [2] * (entities_idx[1] - entities_idx[0] - 1)
    right = [3] * (sen_len - entities_idx[1])

    return left + middle + right


def _pos_feature(sent_len: int, entity_idx: int, entity_len: int,
                 pos_limit: int) -> List:

    left = list(range(-entity_idx, 0))
    middle = [0] * entity_len
    right = list(range(1, sent_len - entity_idx - entity_len + 1))
    pos = left + middle + right

    for i, p in enumerate(pos):
        if p > pos_limit:
            pos[i] = pos_limit
        if p < -pos_limit:
            pos[i] = -pos_limit
    pos = [p + pos_limit + 1 for p in pos]

    return pos


def _build_data(data: List[Dict], vocab: Vocab, relations: Dict) -> List[Dict]:

    if vocab.name == 'LM':
        for d in data:
            d['seq_len'] = len(d['lm_idx'])
            d['target'] = relations[d['relation']]

        return data

    for d in data:
        if vocab.name == 'word':
            word2idx = [vocab.word2idx.get(w, 1) for w in d['words']]
            seq_len = len(word2idx)
            head_idx, tail_idx = d['head_idx'], d['tail_idx']
            head_len, tail_len = 1, 1

        elif vocab.name == 'char':
            word2idx = [
                vocab.word2idx.get(w, 1) for w in d['sentence'].strip()
            ]
            seq_len = len(word2idx)
            head_idx, tail_idx = int(d['head_offset']), int(d['tail_offset'])
            head_len, tail_len = len(d['head']), len(d['tail'])

        entities_idx = [head_idx, tail_idx
                        ] if tail_idx > head_idx else [tail_idx, head_idx]
        head_pos = _pos_feature(seq_len, head_idx, head_len, config.pos_limit)
        tail_pos = _pos_feature(seq_len, tail_idx, tail_len, config.pos_limit)
        mask_pos = _mask_feature(entities_idx, seq_len)
        target = relations[d['relation']]

        d['word2idx'] = word2idx
        d['seq_len'] = seq_len
        d['head_pos'] = head_pos
        d['tail_pos'] = tail_pos
        d['mask_pos'] = mask_pos
        d['target'] = target

    return data


def _build_vocab(data: List[Dict], out_path: Path) -> Vocab:
    if config.word_segment:
        vocab = Vocab('word')
        for d in data:
            vocab.add_sent(d['words'])
    else:
        vocab = Vocab('char')
        for d in data:
            vocab.add_sent(d['sentence'].strip())
    vocab.trim(config.min_freq)

    ensure_dir(out_path)
    vocab_path = os.path.join(out_path, 'vocab.pkl')
    vocab_txt = os.path.join(out_path, 'vocab.txt')
    save_pkl(vocab_path, vocab, 'vocab')
    with open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join([word for word in vocab.word2idx.keys()]))
    return vocab


def _split_sent(data: List[Dict], verbose: bool = True) -> List[Dict]:
    if verbose:
        print('need word segment, use jieba to split sentence')

    jieba.add_word('HEAD')
    jieba.add_word('TAIL')

    for d in data:
        sent = d['sentence'].strip()
        sent = sent.replace(d['head'], 'HEAD', 1)
        sent = sent.replace(d['tail'], 'TAIL', 1)
        sent = jieba.lcut(sent)
        head_idx, tail_idx = sent.index('HEAD'), sent.index('TAIL')
        sent[head_idx], sent[tail_idx] = d['head'], d['tail']
        d['words'] = sent
        d['head_idx'] = head_idx
        d['tail_idx'] = tail_idx
    return data


def _add_lm_data(data: List[Dict]) -> List[Dict]:
    '使用语言模型的词表，序列化输入的句子'
    tokenizer = BertTokenizer.from_pretrained('../bert_pretrained')

    for d in data:
        sent = d['sentence'].strip()
        d['seq_len'] = len(sent)
        sent = sent.replace(d['head'], d['head_type'], 1)
        sent = sent.replace(d['tail'], d['tail_type'], 1)
        sent += '[SEP]' + d['head'] + '[SEP]' + d['tail']
        d['lm_idx'] = tokenizer.encode(sent, add_special_tokens=True)

    return data


def _load_relations(fp: Path) -> Dict:
    '读取关系文件，并将关系保存为词典格式，用来序列化关系'

    print(f'load {fp}')
    relations_arr = []
    relations_dict = {}

    with open(fp, encoding='utf-8') as f:
        for l in f:
            relations_arr.append(l.strip())

    for k, v in enumerate(relations_arr):
        relations_dict[v] = k

    return relations_dict


def process(data_path: Path, out_path: Path) -> None:
    print('===== start preprocess data =====')
    train_fp = os.path.join(data_path, 'train.csv')
    test_fp = os.path.join(data_path, 'test.csv')
    relation_fp = os.path.join(data_path, 'relation.txt')

    print('load raw files...')
    train_raw_data = load_csv(train_fp)
    test_raw_data = load_csv(test_fp)
    relations = _load_relations(relation_fp)

    # 使用预训练语言模型时
    if config.model_name == 'LM':
        print('\nuse pretrained language model serialize sentence...')
        train_raw_data = _add_lm_data(train_raw_data)
        test_raw_data = _add_lm_data(test_raw_data)
        vocab = Vocab('LM')

    else:
        # 当为中文时是否需要分词操作，如果句子已为分词的结果，则不需要分词
        print('\nverify whether need split words...')
        if config.is_chinese and config.word_segment:
            train_raw_data = _split_sent(train_raw_data)
            test_raw_data = _split_sent(test_raw_data, verbose=False)

        print('build word vocabulary...')
        vocab = _build_vocab(train_raw_data, out_path)

    print('\nbuild train data...')
    train_data = _build_data(train_raw_data, vocab, relations)
    print('build test data...\n')
    test_data = _build_data(test_raw_data, vocab, relations)

    ensure_dir(out_path)
    train_data_path = os.path.join(out_path, 'train.pkl')
    test_data_path = os.path.join(out_path, 'test.pkl')

    save_pkl(train_data_path, train_data, 'train data')
    save_pkl(test_data_path, test_data, 'test data')

    print('===== end preprocess data =====')


if __name__ == "__main__":
    data_path = '../data/origin'
    out_path = '../data/out'

    process(data_path, out_path)
