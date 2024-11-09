import random
random.seed(42)
from convert.processer.processer import Processer


class REProcesser(Processer):
    def __init__(self, type_list, role_list, type_role_dict, negative=-1):
        super().__init__(type_list, role_list, type_role_dict, negative)

    def filter_lable(self, record):
        labels = []
        already = set()
        for label in record['relation']:
            if (label['head'], label['relation'], label['tail']) in already:
                continue
            already.add((label['head'], label['relation'], label['tail']))
            labels.append(label)
        record['relation'] = labels
        return record


    def set_negative(self, neg_schema):
        self.label_length = len(self.role_list)
        self.negative = neg_schema


    def get_schemas(self, task_record=None):
        return list(self.role_list)
    

    def get_task_record(self, record):
        return record.get('relation', None)
    

    def get_positive_type_role(self, records):
        positive_role = set()
        for record in records:
            positive_role.add(record['relation'])
        return positive_role, self.role_list