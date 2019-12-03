import os
import csv
import pickle
import logging
from typing import NewType, List, Tuple, Dict, Any

__all__ = [
    'load_pkl',
    'save_pkl',
    'load_csv',
    'save_csv',
]

logger = logging.getLogger(__name__)

Path = str


def load_pkl(fp: Path, verbose: bool = True) -> Any:
    if verbose:
        logger.info(f'load data from {fp}')

    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data: Any, fp: Path, verbose: bool = True) -> None:
    if verbose:
        logger.info(f'save data in {fp}')

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def load_csv(fp: Path, is_tsv: bool = False, verbose: bool = True) -> List:
    if verbose:
        logger.info(f'load csv from {fp}')

    dialect = 'excel-tab' if is_tsv else 'excel'
    with open(fp, encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def save_csv(data: List[Dict], fp: Path, save_in_tsv: False, write_head=True, verbose=True) -> None:
    if verbose:
        logger.info(f'save csv file in: {fp}')

    with open(fp, 'w', encoding='utf-8') as f:
        fieldnames = data[0].keys()
        dialect = 'excel-tab' if save_in_tsv else 'excel'
        writer = csv.DictWriter(f, fieldnames=fieldnames, dialect=dialect)
        if write_head:
            writer.writeheader()
        writer.writerows(data)
