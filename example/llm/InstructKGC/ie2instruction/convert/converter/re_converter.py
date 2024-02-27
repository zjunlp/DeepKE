import json
from collections import defaultdict
from convert.converter.converter import Converter


class REConverter(Converter):
    def __init__(self, language="zh", NAN="NAN", prefix=""):
        super().__init__(language, NAN, prefix)

    

    def get_label_dict(self, records):
        labels = defaultdict(list)
        for rel in records:
            head = rel['head']
            relation = rel['relation']
            tail = rel['tail']
            labels[relation].append({'head':head, 'tail':tail})
        return labels


    def convert_target(self, text, schema, label_dict):
        outputs = defaultdict(list)
        for it in schema:
            if it not in label_dict:
                outputs[it] = []
            else:
                outputs[it] = label_dict[it]
        
        sorted_outputs = {}
        for key, value in outputs.items():
            value_index = []
            for it in value:
                value_index.append([text.find(it['head']), text.find(it['tail']), it])
            value_index.sort(key=lambda x:(x[0], x[1]))
            sorted_value = []
            for it in value_index:
                sorted_value.append(it[1])
            sorted_outputs[key] = value

        output_text = json.dumps(dict(sorted_outputs), ensure_ascii=False)
        return self.prefix + output_text
    
