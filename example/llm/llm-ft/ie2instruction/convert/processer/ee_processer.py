import random
random.seed(42)
import numpy as np
np.random.seed(42)
from collections import defaultdict
from convert.processer.processer import Processer



class EEProcesser(Processer):
    def __init__(self, type_list, role_list, type_role_dict, negative=-1):
        super().__init__(type_list, role_list, type_role_dict, negative)
    
    def filter_lable(self, record):
        return record


    def set_negative(self, neg_schema):
        self.label_length = len(self.type_role_dict)
        self.negative = neg_schema
        self.negative_role = neg_schema


    def get_schemas(self, task_record=None):
        type_role_list = []
        for key, value in self.type_role_dict.items():
            type_role_list.append({'event_type': key, 'trigger':True, 'arguments':list(value)})
        return type_role_list
    

    def get_task_record(self, record):
        return record.get('event', None)
    

    def get_positive_type_role(self, records):
        positive_type = set()
        positive_role = set()
        positive_type_role = defaultdict(set)
        for record in records:
            positive_type.add(record['event_type'])
            for arg in record['arguments']:
                positive_role.add(arg['role'])
                positive_type_role[record['event_type']].add(arg['role'])
        for it in positive_type:
            if it not in positive_type_role:
                positive_type_role[it] = set()
        return positive_type, positive_role, positive_type_role
    

    def get_final_schemas(self, total_schemas):
        final_schemas = []
        for it in total_schemas:
            tmp_schema = []
            for iit in it:
                tmp_schema.append({'event_type':iit[0], 'trigger':True, 'arguments': iit[1]})
            final_schemas.append(tmp_schema)
        return final_schemas
    
    
    def negative_sample(self, record, split_num=4, random_sort=True):
        task_record = self.get_task_record(record)
        # 负采样
        positive_type, _, positive_type_role = self.get_positive_type_role(task_record)
        tmp_total_schemas = []
        if self.negative == 0:   # 不负采样, 全是正样本
            for it in positive_type:
                tmp_total_schemas.append((it, list(self.type_role_dict[it])))
        else:
            negative_type = self.type_list - positive_type
            negative_type = list(negative_type)
            if self.negative > 0: 
                neg_length = sum(np.random.binomial(1, self.negative, self.label_length))
                if len(positive_type) == 0:
                    neg_length = max(neg_length, 1)
                neg_length = min(neg_length, len(negative_type))
                negative_type = random.sample(negative_type, neg_length) 
            
            for it in positive_type:
                negative_role = self.type_role_dict[it] - positive_type_role[it]
                negative_role = list(negative_role)
                if self.negative > 0: 
                    neg_length = sum(np.random.binomial(1, self.negative_role, len(self.type_role_dict[it])))
                    if len(positive_type_role[it]) == 0:
                        neg_length = max(neg_length, 1)
                    neg_length = min(neg_length, len(negative_role))
                    negative_role = random.sample(negative_role, neg_length)
                tmp_total_schemas.append((it, negative_role + list(positive_type_role[it])))
            for it in negative_type:
                negative_role = list(self.type_role_dict[it])
                if self.negative > 0:
                    neg_length = sum(np.random.binomial(1, self.negative_role, len(self.type_role_dict[it])))
                    if len(positive_type_role[it]) == 0:
                        neg_length = max(neg_length, 1)
                    neg_length = min(neg_length, len(negative_role))
                    negative_role = random.sample(negative_role, neg_length)
                tmp_total_schemas.append((it, negative_role))

        # 排序
        if not random_sort:
            sorted_tmp_total_schemas = sorted(tmp_total_schemas, key=lambda x: (x[0], x[1]))
        else:
            random.shuffle(tmp_total_schemas)
            sorted_tmp_total_schemas = []
            for it in tmp_total_schemas:
                random.shuffle(it[1])
                sorted_tmp_total_schemas.append(it)

        # 按照split_num划分
        return self.split_total_by_num(split_num, sorted_tmp_total_schemas)

    