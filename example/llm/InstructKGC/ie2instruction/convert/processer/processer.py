import os
import numpy as np
np.random.seed(42)
import json
from collections import defaultdict
import random
random.seed(42)



class Processer:
    def __init__(self, type_list, role_list, type_role_dict, negative=-1):
        self.type_list = set(type_list)
        self.role_list = set(role_list)
        self.type_role_dict = defaultdict(set)
        for key, value in type_role_dict.items():
            self.type_role_dict[key] = set(value)
        self.negative = negative

    
    def filter_lable(self, record):
        raise NotImplementedError
    

    def set_negative(self, neg_schema):
        raise NotImplementedError
    

    def read_data(self, path):
        datas = []
        with open(path, 'r', encoding='utf-8') as reader:
            for line in reader:
                data = json.loads(line)
                data = self.filter_lable(data)
                if data is None:
                    continue
                datas.append(data)
        return datas


    @staticmethod
    def read_from_file(processer_class, filename, negative=3):
        if os.path.exists(filename) == False:
            return processer_class(set(), set(), defaultdict(set), negative)
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return processer_class(type_list, role_list, type_role_dict, negative)


    def get_schemas(self, task_record=None):
        raise NotImplementedError
    

    def get_task_record(self, record):
        raise NotImplementedError
    

    def get_positive_type_role(self, records):
        raise NotImplementedError
    

    def get_final_schemas(self, total_schemas):
        return total_schemas
    

    def split_total_by_num(self,split_num, tmp_total_schemas):
        if split_num == -1:
            total_schemas = [tmp_total_schemas, ]
        else:
            negative_length = max(len(tmp_total_schemas) // split_num, 1) * split_num
            total_schemas = []
            for i in range(0, negative_length, split_num):
                total_schemas.append(tmp_total_schemas[i:i+split_num])
            # 剩余的不足split_num的部分
            remain_len = max(1, split_num // 2)
            if len(tmp_total_schemas) - negative_length >= remain_len:
                total_schemas.append(tmp_total_schemas[negative_length:])
            else:
                total_schemas[-1].extend(tmp_total_schemas[negative_length:])
        return self.get_final_schemas(total_schemas)


    def negative_sample(self, record, split_num=4, random_sort=True):
        task_record = self.get_task_record(record)
        # 负采样
        positive, type_role = self.get_positive_type_role(task_record)
        if self.negative == 0:   # 不负采样, 全是正样本
            if len(positive) == 0:
                return []
            tmp_total_schemas = list(positive)
        else:
            negative = type_role - positive
            negative = list(negative)
            if self.negative > 0: 
                neg_length = sum(np.random.binomial(1, self.negative, self.label_length))
                if len(positive) == 0:
                    neg_length = max(neg_length, 1)
                neg_length = min(neg_length, len(negative))
                negative = random.sample(negative, neg_length) 
            tmp_total_schemas = negative + list(positive)

        # 排序
        if not random_sort:
            tmp_total_schemas = sorted(tmp_total_schemas)
        else:
            random.shuffle(tmp_total_schemas)

        # 按照split_num划分
        return self.split_total_by_num(split_num, tmp_total_schemas)

