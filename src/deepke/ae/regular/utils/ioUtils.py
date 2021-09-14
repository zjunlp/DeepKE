import os
import csv
import json
import pickle
import logging
from typing import NewType, List, Tuple, Dict, Any

__all__ = [
    'load_pkl',
    'save_pkl',
    'load_csv',
    'save_csv',
    'load_jsonld',
    'save_jsonld',
    'jsonld2csv',
    'csv2jsonld',
]

logger = logging.getLogger(__name__)

Path = str


def load_pkl(fp: Path, verbose: bool = True) -> Any:
    """
    读取文件
    Args :
        fp (String) : 读取数据地址
        verbose (bool) : 是否打印日志
    Return : 
        data (Any) : 读取的数据
    """
    if verbose:
        logger.info(f'load data from {fp}')

    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data: Any, fp: Path, verbose: bool = True) -> None:
    """
    保存文件
    Args :
        data (Any) : 数据
        fp (String) :保存的地址
        verbose (bool) : 是否打印日志
    """
    if verbose:
        logger.info(f'save data in {fp}')

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def load_csv(fp: Path, is_tsv: bool = False, verbose: bool = True) -> List:
    """
    读取csv格式文件
    Args :
        fp (String) : 保存地址
        is_tsv (bool) : 是否为excel-tab格式
        verbose (bool) : 是否打印日志
    Return :
        list(reader) (List): 读取的List数据
    """
    if verbose:
        logger.info(f'load csv from {fp}')

    dialect = 'excel-tab' if is_tsv else 'excel'
    with open(fp, encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def save_csv(data: List[Dict], fp: Path, save_in_tsv: False, write_head=True, verbose=True) -> None:
    """
    保存csv格式文件
    Args :
        data (List) : 所需保存的List数据
        fp (String) : 保存地址
        save_in_tsv (bool) : 是否保存为excel-tab格式
        write_head (bool) : 是否写表头
        verbose (bool) : 是否打印日志
    """
    if verbose:
        logger.info(f'save csv file in: {fp}')

    with open(fp, 'w', encoding='utf-8') as f:
        fieldnames = data[0].keys()
        dialect = 'excel-tab' if save_in_tsv else 'excel'
        writer = csv.DictWriter(f, fieldnames=fieldnames, dialect=dialect)
        if write_head:
            writer.writeheader()
        writer.writerows(data)


def load_jsonld(fp: Path, verbose: bool = True) -> List:
   
    """
    读取jsonld文件
    Args:
        fp (String): jsonld 文件地址
        verbose (bool): 是否打印日志
    Return: 
        datas (List) : 读取后的List
    """
    if verbose:
        logger.info(f'load jsonld from {fp}')

    datas = []
    with open(fp, encoding='utf-8') as f:
        for l in f:
            line = json.loads(l)
            data = list(line.values())
            datas.append(data)

    return datas


def save_jsonld(fp):
    """
    保存jsonld格式文件
    """
    pass


def jsonld2csv(fp: str, verbose: bool = True) -> str:
    """
    读入 jsonld 文件，存储在同位置同名的 csv 文件
    Args:
        fp (String): jsonld 文件地址
        verbose (bool): 是否打印日志
    Return: 
        fp_new (String):文件地址
    """
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
    """
    读入 csv 文件，存储在同位置同名的 jsonld 文件
    Args:
        fp (String): csv 文件地址
        verbose (bool): 是否打印日志
    Return: 
        fp_new (String):文件地址
    """
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
        f.write(os.linesep.join([json.dumps(l, ensure_ascii=False) for l in data]))
    if verbose:
        print(f'saved jsonld file in: {fp_new}')
    return fp_new
