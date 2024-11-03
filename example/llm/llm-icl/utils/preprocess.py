import os
import json


def prepare_examples(
        task,
        language,
        data_path: str = "data"
):
    """
        :param data_path: (str) 数据文件所在的目录路径，包含任务和语言特定的 JSON 文件。
        :param task: (str) 任务类型，包括 'ner'（命名实体识别）、're'（关系抽取）、'da'（对话生成）、'ee'（事件抽取）和 'rte'（推理任务）。
        :param language: (str) 语言标识符，用于指定数据文件的语言（例如 'zh' 表示中文）。
        :return: (list) 返回一个包含准备好的示例数据的字典列表，每个字典对应一个示例，格式根据任务类型不同而不同。
    """
    data_name = task + '_' + language + '.json'
    data_path = os.path.join(data_path, data_name)

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
    print(prepare_examples(data_path='../data', task='da', language='ch'))
    print(prepare_examples(data_path='../data', task='ee', language='en'))
    print(prepare_examples(data_path='../data', task='ner', language='ch'))
    print(prepare_examples(data_path='../data', task='re', language='en'))
    print(prepare_examples(data_path='../data', task='rte', language='ch'))

