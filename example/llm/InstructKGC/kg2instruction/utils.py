import random
import json
import hashlib


def stable_hash(input_str):
    sha256 = hashlib.sha256()
    sha256.update(input_str.encode('utf-8'))
    return sha256.hexdigest()


def match_sublist(the_list, to_match):
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match)]
    return matched_list


def rel_sort(text, rels):
    new_rels = []
    rels_mapper = {}
    for rel in rels:
        if rel['relation'] not in rels_mapper:
            type_id = random.randint(0, 1000)
            rels_mapper[rel['relation']] = type_id
        else:
            type_id = rels_mapper[rel['relation']] 
        head_offset = match_sublist(list(text), list(rel['head']))
        tail_offset = match_sublist(list(text), list(rel['tail']))
        if len(head_offset) == 0 or len(tail_offset) == 0:
            continue
        head_offset = head_offset[0]
        tail_offset = tail_offset[0]
        new_rels.append([rel, [head_offset[0], type_id, tail_offset[0]]])
    new_rels = sorted(new_rels, key=lambda x: (x[1][1], x[1][0], x[1][2]))
    new_rels = [it[0] for it in new_rels]
    rels_list = sorted(rels_mapper.items(), key=lambda x: x[1])
    rels_list = [it[0] for it in rels_list]
    return new_rels, rels_list


def ent_sort(text, ents):
    new_ents = []
    ents_mapper = {}
    for ent in ents:
        if ent['entity_type'] not in ents_mapper:
            type_id = random.randint(0, 1000)
            ents_mapper[ent['entity_type']] = type_id
        else:
            type_id = ents_mapper[ent['entity_type']] 
        ent_offset = match_sublist(list(text), list(ent['entity']))
        if len(ent_offset) == 0:
            continue
        ent_offset = ent_offset[0]
        new_ents.append([ent, [ent_offset[0], type_id]])
    new_ents = sorted(new_ents, key=lambda x: (x[1][1], x[1][0]))
    new_ents = [it[0] for it in new_ents]
    ents_list = sorted(ents_mapper.items(), key=lambda x: x[1])
    ents_list = [it[0] for it in ents_list]
    return new_ents, ents_list


def get_type(records, task):
    ll = set()
    if task == "NER":
        for record in records:
            ll.add(record['entity_type'])
    elif task == "RE":
        for record in records:
            ll.add(record['relation'])
    elif task == "EE":
        for record in records:
            ll.add(record['event_type'])
    return ll


class FullSampler:
    def __init__(self, type_list, role_list, type_role_dict=None):
        self.type_list = set(type_list)
        self.role_list = set(role_list)
        self.type_role_dict = type_role_dict

    def negative_sample(self, record, task):
        positive = get_type(record, task)
        negative = list()
        if task == "RE":
            negative = self.role_list - positive
            for it in negative:
                record.append({"head":"", "relation":it, "tail":""})
        elif task == "NER":
            negative = self.type_list - positive
            for it in negative:
                record.append({"entity":"", "entity_type":it})
        else:
            negative = self.type_list - positive
            for it in negative:
                record.append({"event_trigger":"", "event_type":it, "arguments":list()})
        type_list = []
        if len(positive) != 0:
            type_list.extend(list(positive))
        if len(negative) != 0:
            type_list.extend(negative)
        return record, type_list, list(self.role_list)


    @staticmethod
    def read_from_file(filename):
        '''
        对于NER任务
        []    # 实体类型列表
        []    # 空列表
        {}    # 空字典
        对于RE任务
        []    # 空列表
        []    # 关系类型列表
        {}    # 空字典
        对于EE任务
        []    # 事件类型列表
        []    # 论元角色列表
        {}    # 空字典 
        '''
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return FullSampler(type_list, role_list, type_role_dict)
