"""
preprocess.py

Mainly prepares example data for those tasks.
"""

import os
import json


def prepare_examples(
        task,
        language,
        data_path: str = None
):
    """
    This function loads data from a JSON file corresponding to the given task
    and language. It processes the data into a list of dictionaries, where each
    dictionary represents an example formatted according to the task type.

    :param task: (str) The type of task, which can be 'ner' (Named Entity Recognition),
    're' (Relation Extraction), 'da' (Dialogue Generation),
    'ee' (Event Extraction), or 'rte' (Reasoning Task).
    :param language: (str) The language identifier used to specify the language of
    the data file (e.g., 'zh' for Chinese).
    :param data_path: (str) The directory path where the data files are located,
    containing task and language-specific JSON files.
    Defaults to "data".
    :return: (list) A list of dictionaries containing prepared example data,
    with each dictionary corresponding to a specific example.
    The format of each dictionary varies depending on the task type.
    """
    if data_path is None:
        data_path = os.path.join(os.getcwd(), "data/icl")

    data_name = task + '_' + language + '.json'
    data_path = os.path.join(data_path, data_name)
    # print(data_path)

    data = json.load(open(data_path, 'r'))
    if task == 'ner':
        data = data['samples']
    # print(data)

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
            example['input'] = item['text']
            example['output'] = item['data']
        elif task == 'ee':
            example['input'] = item['text']
            example['output'] = item['event_list']
        elif task == 'rte':
            example['input'] = item['text']
            example['output'] = item['labels']
        examples.append(example)

    return examples


if __name__ == "__main__":
    # test:
    current_data_path = os.path.join(os.path.dirname(__file__), "../../data/icl")
    print(prepare_examples(data_path=current_data_path, task='da', language='ch'))
    print(prepare_examples(data_path=current_data_path, task='ee', language='en'))
    print(prepare_examples(data_path=current_data_path, task='ner', language='ch'))
    print(prepare_examples(data_path=current_data_path, task='re', language='en'))
    print(prepare_examples(data_path=current_data_path, task='rte', language='ch'))
