import argparse
import json
from collections import Counter, defaultdict, OrderedDict
import ipdb
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", type=str, required=True)
args = parser.parse_args()

data = []
for line in open(args.input_path, 'r', encoding='utf-8'):
    data.append(json.loads(line))

class Info:
    def __init__(self):
        self.sentence_num = 0
        self.entity = Counter()
        self.entity_num = 0
        self.event = Counter()
        self.event_num = 0
        self.role = Counter()
        self.role_num = 0
    
    def update(self, instance):
        self.sentence_num += 1
        id2entity = dict()
        for entity in instance['entity_mentions']:
            id2entity[entity['id']] = entity
            self.entity[entity['entity_type']] += 1
            self.entity_num += 1
        
        for event in instance['event_mentions']:
            self.event[event['event_type']] += 1
            self.event_num += 1
            for argument in event['arguments']:
                self.role[argument['role']] += 1
                self.role_num += 1
    
    def __add__(self, obj):
        self.sentence_num += obj.sentence_num
        self.entity = self.entity + obj.entity
        self.entity_num += obj.entity_num
        self.event = self.event + obj.event
        self.event_num += obj.event_num
        self.role = self.role + obj.role
        self.role_num += obj.role_num
        return self

def get_statistics(list_of_key, d_stat):
    total_sent = 0
    total_event = Counter()
    total_event_num = 0
    total_role = Counter()
    total_role_num = 0
    total_entity_num = 0
    for l in list_of_key:
        assert l in d_stat.keys()
        total_sent += d_stat[l].sentence_num
        total_event += d_stat[l].event
        total_role += d_stat[l].role
        total_event_num += d_stat[l].event_num
        total_role_num += d_stat[l].role_num
        total_entity_num += d_stat[l].entity_num

    return len(list_of_key), total_sent, total_event_num, len(total_event), total_role_num, len(total_role)

def aggregate(list_of_key, d_stat):
    aggre = Info()
    for l in list_of_key:
        assert l in d_stat.keys()
        aggre = aggre+d_stat[l]
    return aggre

# get info
doc_statistics = defaultdict(Info)
for instance in data:
    info = doc_statistics[instance['doc_id']]
    info.update(instance)

def export_doc_list(lists, filename):
    with open(filename, 'w') as f:
        for l in lists:
            f.write(l + '\n')

def read_doc_list(filename):
    lists = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            lists.append(l.strip('\n'))
    return lists