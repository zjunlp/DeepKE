import random
random.seed(42)
from convert.processer.processer import Processer


class EETProcesser(Processer):
    def __init__(self, type_list, role_list, type_role_dict, negative=-1):
        super().__init__(type_list, role_list, type_role_dict, negative)


    def filter_lable(self, record):
        return record


    def set_negative(self, neg_schema):
        self.label_length = len(self.type_list)
        self.negative = neg_schema


    def get_schemas(self, task_record=None):
        return list(self.type_list)
    

    def get_task_record(self, record):
        return record.get('event', None)
        

    def get_positive_type_role(self, records):
        positive_type = set()
        for record in records:
            positive_type.add(record['event_type'])
        return positive_type, self.type_list