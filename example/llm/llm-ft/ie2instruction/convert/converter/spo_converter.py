import json
from collections import defaultdict
from convert.converter.converter import Converter
from convert.utils.utils import schema2spotext

class SPOConverter(Converter):
    def __init__(self, language="zh", NAN="NAN", prefix=""):
        super().__init__(language, NAN, prefix)

    
    def get_label_dict(self, records):
        labels = defaultdict(list)
        for rel in records:
            head = rel['head']
            head_type = rel['head_type']
            relation = rel['relation']
            tail = rel['tail']
            tail_type = rel['tail_type']
            spotext = schema2spotext(head_type, relation, tail_type)
            labels[spotext].append({'subject':head, 'object':tail})
        return labels
    
    def convert_target(self, text, schema, label_dict):            
        outputs = defaultdict(list)
        for it in schema:
            spotext = schema2spotext(it['subject_type'], it['predicate'], it['object_type'])
            relation = it['predicate']
            if spotext not in label_dict:
                outputs[relation] = []
            else:
                outputs[relation] = label_dict[spotext]
                
        sorted_outputs = {}
        for key, value in outputs.items():
            value_index = []
            for it in value:
                value_index.append([text.find(it['subject']), text.find(it['object']), it])
            value_index.sort(key=lambda x:(x[0], x[1]))
            sorted_value = []
            for it in value_index:
                sorted_value.append(it[1])
            sorted_outputs[key] = value

        output_text = json.dumps(dict(sorted_outputs), ensure_ascii=False)
        return self.prefix + output_text
    