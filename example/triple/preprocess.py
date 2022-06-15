import os
import logging
from collections import OrderedDict
from typing import List, Dict
from transformers import BertTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

logger = logging.getLogger(__name__)


__all__ = [
    "_lm_serialize",
    "_handle_relation_data",
]
def _lm_serialize(data: List[Dict], cfg):
    logger.info('use bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm_file)
    for d in data:
        sent = d['sentence'].strip()
        sent = sent.replace(d['head'], d['head_type'], 1).replace(d['tail'], d['tail_type'], 1)
        sent += '[SEP]' + d['head'] + '[SEP]' + d['tail']
        d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)
        d['seq_len'] = len(d['token2idx'])



def _handle_relation_data(relation_data: List[Dict]) -> Dict:
    rels = OrderedDict()
    relation_data = sorted(relation_data, key=lambda i: int(i['index']))
    for d in relation_data:
        rels[d['relation']] = {
            'index': int(d['index']),
            'head_type': d['head_type'],
            'tail_type': d['tail_type'],
        }

    return rels

