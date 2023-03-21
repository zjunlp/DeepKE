import os
import json


def prepare_examples(data_path, task, language):
    data_name = task + '_' + language + '.json'
    data_path = os.path.join(data_path, data_name)
    data = json.load(open(data_path, 'r'))
    if task == 'ner':
        data = data['samples']
    examples = []
    for item in data:
        example = {}
        if task in ['re', 'da']:
            example['context'] = item['text']
            example['head_type'] = item['head_type']
            example['head_entity'] = item['head_entity']
            example['tail_type'] = item['tail_type']
            example['tail_entity'] = item['tail_entity']
            example['relation'] = item['relation']
        elif task == 'ner':
            example['output'] = item['data']
        elif task == 'ee':
            example['input'] = item['text']
            example['output'] = item['event_list']
        elif task == 'rte':
            example['input'] = item['text']
            example['output'] = item['labels']
        examples.append(example)
    return examples