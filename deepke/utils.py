import os
import csv
import json
import torch
import pickle
import random
import warnings
import numpy as np
from functools import reduce
from typing import Dict, List, Tuple, Set, Any

__all__ = [
    'to_one_hot',
    'seq_len_to_mask',
    'ignore_waring',
    'make_seed',
    'load_pkl',
    'save_pkl',
    'ensure_dir',
    'load_csv',
    'load_jsonld',
    'jsonld2csv',
    'csv2jsonld',
]

Path = str

def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length).to(x.device)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot


def model_summary(model):
    """
    得到模型的总参数量

    :params model: Pytorch 模型
    :return tuple: 包含总参数量，可训练参数量，不可训练参数量
    """
    train = []
    nontrain = []

    def layer_summary(module):
        def count_size(sizes):
            return reduce(lambda x, y: x * y, sizes)

        for p in module.parameters(recurse=False):
            if p.requires_grad:
                train.append(count_size(p.shape))
            else:
                nontrain.append(count_size(p.shape))
        for subm in module.children():
            layer_summary(subm)

    layer_summary(model)
    total_train = sum(train)
    total_nontrain = sum(nontrain)
    total = total_train + total_nontrain
    strings = []
    strings.append('Total params: {:,}'.format(total))
    strings.append('Trainable params: {:,}'.format(total_train))
    strings.append('Non-trainable params: {:,}'.format(total_nontrain))
    max_len = len(max(strings, key=len))
    bar = '-' * (max_len + 3)
    strings = [bar] + strings + [bar]
    print('\n'.join(strings))
    return total, total_train, total_nontrain


def seq_len_to_mask(seq_len, max_len=None):
    """

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(
            np.shape(seq_len)
        ) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim(
        ) == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size,
                                                          -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


def ignore_waring():
    warnings.filterwarnings("ignore")


def make_seed(num: int = 1) -> None:
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)


def load_pkl(fp: str, obj_name: str = 'data', verbose: bool = True) -> Any:
    if verbose:
        print(f'load {obj_name} in {fp}')
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(fp: Path, obj, obj_name: str = 'data',
             verbose: bool = True) -> None:
    if verbose:
        print(f'save {obj_name} in {fp}')
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def ensure_dir(d: str, verbose: bool = True) -> None:
    '''
    判断目录是否存在，不存在时创建
    :param d: directory
    :param verbose: whether print logging
    :return: None
    '''
    if not os.path.exists(d):
        if verbose:
            print("Directory '{}' do not exist; creating...".format(d))
        os.makedirs(d)


def load_csv(fp: str) -> List:
    print(f'load {fp}')

    with open(fp, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)



def load_jsonld(fp: str) -> List:
    print(f'load {fp}')
    datas = []

    with open(fp, encoding='utf-8') as f:
        for l in f:
            line = json.loads(l)
            data = list(line.values())
            datas.append(data)
    return datas


def jsonld2csv(fp: str, verbose: bool = True) -> str:
    '''
    读入 jsonld 文件，存储在同位置同名的 csv 文件
    :param fp: jsonld 文件地址
    :param verbose: whether print logging
    :return: csv 文件地址
    '''
    data = []
    root, ext = os.path.splitext(fp)
    fp_new = root + '.csv'
    if verbose:
        print(f'read jsonld file in: {fp}')
    with open(fp, encoding='utf-8') as f:
        for l in f:
            line = json.loads(l)
            data.append(line)
    if verbose:
        print('saving...')
    with open(fp_new, 'w', encoding='utf-8') as f:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames, dialect='excel')
        writer.writeheader()
        writer.writerows(data)
    if verbose:
        print(f'saved csv file in: {fp_new}')
    return fp_new


def csv2jsonld(fp: str, verbose: bool = True) -> str:
    '''
    读入 csv 文件，存储为同位置同名的 jsonld 文件
    :param fp: csv 文件地址
    :param verbose: whether print logging
    :return: jsonld 地址
    '''
    data = []
    root, ext = os.path.splitext(fp)
    fp_new = root + '.jsonld'
    if verbose:
        print(f'read csv file in: {fp}')
    with open(fp, encoding='utf-8') as f:
        writer = csv.DictReader(f, fieldnames=None, dialect='excel')
        for line in writer:
            data.append(line)
    if verbose:
        print('saving...')
    with open(fp_new, 'w', encoding='utf-8') as f:
        f.write(
            os.linesep.join([json.dumps(l, ensure_ascii=False) for l in data]))
    if verbose:
        print(f'saved jsonld file in: {fp_new}')
    return fp_new


if __name__ == '__main__':
    pass
