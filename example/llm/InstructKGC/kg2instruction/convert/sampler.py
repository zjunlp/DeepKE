import json
import os
from collections import defaultdict
import random
random.seed(42)

def get_positive_type_role(records, task):
    positive_type = set()
    positive_role = set()
    positive_type_role = defaultdict(set)
    if task == "NER":
        for record in records:
            positive_type.add(record['entity_type'])
    elif task == "RE":
        for record in records:
            positive_role.add(record['relation'])
    elif task == "EE":
        for record in records:
            positive_type.add(record['event_type'])
            for arg in record['arguments']:
                positive_role.add(arg['role'])
                positive_type_role[record['event_type']].add(arg['role'])
        for it in positive_type:
            if it not in positive_type_role:
                positive_type_role[it] = set()
    elif task == "EET":
        for record in records:
            positive_type.add(record['event_type'])
    return positive_type, positive_role, positive_type_role



class Sampler:
    def __init__(self, type_list, role_list, type_role_dict, negative=3):
        self.type_list = set(type_list)
        self.role_list = set(role_list)
        self.type_role_dict = defaultdict(set)
        for key, value in type_role_dict.items():
            self.type_role_dict[key] = set(value)
        self.negative = negative

    def set_negative(self, negative):
        self.negative = negative

    def negative_sample(self, record, task):
        positive_type, positive_role, positive_type_role = get_positive_type_role(record, task)
        negative = list()
        if task == "RE":
            negative = self.role_list - positive_role    # 负样本
            if self.negative > 0:     # <0, 是采样所有的负样本
                negative = random.sample(self.role_list, self.negative)
            for it in negative:
                if it not in positive_role:
                    record.append({"head":"", "relation":it, "tail":""})
        elif task == "NER":
            negative = self.type_list - positive_type
            if self.negative > 0:     # <0, 是采样所有的负样本
                negative = random.sample(self.type_list, self.negative)
            for it in negative:
                if it not in positive_type:
                    record.append({"entity":"", "entity_type":it})
        elif task == "EE":
            negative_type = self.type_list - positive_type
            negative_type_role = defaultdict(set)
            if self.negative > 0:
                for key, value in positive_type_role.items():
                    negative_role = self.type_role_dict[key] - value
                    if len(negative_role) > 0:
                        negative_type_role[key] = negative_role
                for event in record:
                    if event['event_type'] in negative_type_role:
                        num_t = min(len(self.type_role_dict[event['event_type']]), self.negative)
                        num_t = random.randint(1, num_t)
                        negative_role = random.sample(self.type_role_dict[event['event_type']], num_t)
                        for it in negative_role:
                            if it not in positive_type_role[event['event_type']]:
                                event['arguments'].append({"argument":"", "role":it})
                                positive_type_role[event['event_type']].add(it)
                negative_type = random.sample(self.type_list, self.negative)
                for it in negative_type:
                    if it not in positive_type:
                        record.append({"event_trigger":"", "event_type":it, "arguments":[{"argument":"", "role":iit} for iit in self.type_role_dict[it]]})
            else:       # <0, 是采样所有的负样本
                for event in record:
                    neg_role = self.type_role_dict[event['event_type']] - positive_type_role[event['event_type']]
                    if len(neg_role) > 0:
                        for it in neg_role:
                            event['arguments'].append({"argument":"", "role":it})
                neg_type = self.type_list - positive_type
                for it in neg_type:
                    record.append({"event_trigger":"", "event_type":it, "arguments":[{"argument":"", "role":iit} for iit in self.type_role_dict[it]]})
        elif task == "EEA":
            negative_type = self.type_list - positive_type
            negative_type_role = defaultdict(set)
            if self.negative > 0:
                for key, value in positive_type_role.items():
                    negative_role = self.type_role_dict[key] - value
                    if len(negative_role) > 0:
                        negative_type_role[key] = negative_role
                for event in record:
                    if event['event_type'] in negative_type_role:
                        num_t = min(len(self.type_role_dict[event['event_type']]), self.negative)
                        num_t = random.randint(1, num_t)
                        negative_role = random.sample(self.type_role_dict[event['event_type']], num_t)
                        for it in negative_role:
                            if it not in positive_type_role[event['event_type']]:
                                event['arguments'].append({"argument":"", "role":it})
                                positive_type_role[event['event_type']].add(it)
            else:          # <0, 是采样所有的负样本
                for event in record:
                    neg_role = self.type_role_dict[event['event_type']] - positive_type_role[event['event_type']]
                    if len(neg_role) > 0:
                        for it in neg_role:
                            event['arguments'].append({"argument":"", "role":it})
        elif task == "EET": 
            negative = self.type_list - positive_type
            if self.negative > 0:
                negative = random.sample(self.type_list, self.negative)
            for it in negative:    # <0, 是采样所有的负样本
                if it not in positive_type:
                    record.append({"event_trigger":"", "event_type":it})
        return record


    @staticmethod
    def read_from_file(filename, negative=3):
        if os.path.exists(filename) == False:
            return Sampler(set(), set(), defaultdict(set), negative)
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return Sampler(type_list, role_list, type_role_dict, negative)


