import json
import os.path
import sys
from copy import deepcopy
from datetime import datetime
from os import path

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from tqdm import tqdm



class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list, verbose=False):
        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)

def init_es():
    return Elasticsearch()

def create_index(es, name):
    index_mappings = {
        "mappings": {
            "properties": {
                "key": {
                    "type": "text"
                },
                "value": {
                    "type": "text"
                },
                "sent": {
                    "type": "text"
                },
                "source": {
                    "type": "text"
                }
            }
        }
    }
    if es.indices.exists(index=name) is not True:
        print("create", name)
        es.indices.create(index=name, body=index_mappings)

def del_index(es, index_name):
    es.indices.delete(index=index_name)


def add_data(es, data, index_name):
    index = 0
    actions = []
    print('---adding data---')
    for key in tqdm(data.keys()):
        item = {
            "_index": index_name,
            "_id": index,
            "_source": {
                "key": key,
                "trigger": data[key][0],
                "event_type": data[key][1],
                "schema": data[key][2]
            }
        }
        actions.append(item)
        index += 1
        if len(actions) == 1000:
            res, _ = helpers.bulk(es, actions)
            print(res)
            del actions[0:len(actions)]

    if index > 0:
        helpers.bulk(es, actions)
    print('---ending data---')

def search(es, query, index_name):
    query_contains = {
        'query': {
            'match': {
                'key': query,
            }
        }
    }
    searched = es.search(index=index_name, body=query_contains, size=32)
    return searched