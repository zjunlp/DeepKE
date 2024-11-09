import random
random.seed(42)
from convert.processer.processer import Processer
from convert.utils.utils import spotext2schema, schema2spotext

class SPOProcesser(Processer):
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
        self.label_length = len(self.type_list)
        self.negative = neg_schema


    def get_schemas(self, task_record=None):
        spo_type = [spotext2schema(it) for it in self.type_list]
        spo_type = [{'subject_type':it[0], 'predicate':it[1], 'object_type':it[2]} for it in spo_type]
        return spo_type
    

    def get_task_record(self, record):
        return record.get('relation', None)

    def get_positive_type_role(self, records):
        positive_type = set()
        for record in records:
            positive_type.add(schema2spotext(record['head_type'], record['relation'], record['tail_type']))
        return positive_type, self.type_list
    

    def get_final_schemas(self, total_schemas):
        final_schemas = []
        for it in total_schemas:
            tmp_schema = []
            for iit in it:
                head_type, relation, tail_type = spotext2schema(iit)
                tmp_schema.append({'subject_type':head_type, 'predicate':relation, 'object_type':tail_type})
            final_schemas.append(tmp_schema)
        return final_schemas


   