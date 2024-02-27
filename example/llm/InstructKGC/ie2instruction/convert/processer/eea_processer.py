import random
random.seed(42)
import numpy as np
np.random.seed(42)
from collections import defaultdict
from convert.processer.ee_processer import EEProcesser


class EEAProcesser(EEProcesser):
    def __init__(self, type_list, role_list, type_role_dict, negative=-1):
        super().__init__(type_list, role_list, type_role_dict, negative)
        

    def filter_lable(self, record):
        if len(record['event']) == 0:
            return None
        return record
    

    def get_schemas(self, task_record=None):
        if len(task_record) == 0:
            return None
        type_role_list = []
        event_types = set()
        for event in task_record:
            event_types.add(event['event_type'])
        for key in event_types:
            type_role_list.append({'event_type': key, 'arguments':list(self.type_role_dict[key])})
        return type_role_list


    def get_task_record(self, record):
        return record.get('event', None)
    

    def get_final_schemas(self, total_schemas):
        final_total_schemas = []
        for it in total_schemas:
            tmp_schema = []
            for iit in it:
                tmp_schema.append({'event_type':iit[0], 'trigger':iit[1], 'arguments': iit[2]})
            final_total_schemas.append(tmp_schema)
        return final_total_schemas
    
    
    def negative_sample(self, record, split_num=4, random_sort=True):
        task_record = self.get_task_record(record)
        label_dict = defaultdict(list)
        for record in task_record:
            label_dict[record['event_type']].append(record['event_trigger'])

        tmp_total_schemas = []
        for key, value in label_dict.items():
            negative_role = list(self.type_role_dict[key])
            if self.negative > 0:
                neg_length = sum(np.random.binomial(1, self.negative_role, len(self.type_role_dict[key])))
                neg_length = max(neg_length, 1)
                neg_length = min(neg_length, len(negative_role))
                negative_role = random.sample(negative_role, neg_length)
            
            neg_length = sum(np.random.binomial(1, self.negative, len(value)))
            neg_length = max(neg_length, 1)
            neg_length = min(neg_length, len(negative_role))
            value = random.sample(value, neg_length)

            tmp_total_schemas.append((key, value, negative_role))

        # 排序
        if not random_sort:
            sorted_tmp_total_schemas = sorted(tmp_total_schemas, key=lambda x: x[0])
        else:
            random.shuffle(tmp_total_schemas)
            sorted_tmp_total_schemas = []
            for it in tmp_total_schemas:
                random.shuffle(it[2])
                sorted_tmp_total_schemas.append(it)

        # 按照split_num划分
        return self.split_total_by_num(split_num, sorted_tmp_total_schemas)
    
