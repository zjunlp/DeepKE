import json
from collections import defaultdict
from convert.converter.converter import Converter

class EETConverter(Converter):
    def __init__(self, language="zh", NAN="", prefix=""):
        super().__init__(language, NAN, prefix)


    def get_label_dict(self, records):
        labels = defaultdict(list)
        for event in records:
            event_trigger = event['event_trigger']
            event_type = event['event_type']
            labels[event_type].append(event_trigger)
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
                value_index.append([text.find(it), it])
            value_index.sort(key=lambda x:x[0])
            sorted_value = []
            for it in value_index:
                sorted_value.append(it[1])
            sorted_outputs[key] = value

        output_text = json.dumps(sorted_outputs, ensure_ascii=False)
        return self.prefix + output_text
    