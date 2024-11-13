"""
preprocess.py

Mainly prepares example data for those tasks.
"""

import os
import json


# 注释全部改为英文，再加一个函数注释：
# 这个函数用于将data_path路径中的train.json文件读取其中的instruction字段并合成。
def prepare_input_plus(
        data_path: str = None,
        language: str = 'en'
) -> str:
    # 定义中文和英文的前缀
    prefix_ch = ""
    prefix_en = ""

    # 根据语言选择前缀
    prefix = prefix_ch if language == 'ch' else prefix_en

    # 读取文件中的所有数据
    with open(data_path, 'r', encoding='utf-8') as file:
        json_lines = [line.strip() for line in file.readlines()]

    prompts = []
    first_instruction = None  # 用于存储第一条数据的 instruction

    for idx, line in enumerate(json_lines):
        # 解析每一行的 JSON 数据
        entry = json.loads(line)
        instruction_str = entry.get('instruction', '')

        # 提取 instruction 中的 JSON 内容
        instruction_json = json.loads(instruction_str)

        instruction = instruction_json.get('instruction', '')
        schema = instruction_json.get('schema', [])
        input_text = instruction_json.get('input', '')

        # 处理第一条数据的 instruction
        if idx == 0:
            first_instruction = instruction

        # 根据 language 构造不同的 prompt
        prompt = f"序号 {idx + 1}: {input_text}\nSchema: {', '.join(schema)}" if language == 'ch' else f"Task {idx + 1}: {input_text}\nSchema: {', '.join(schema)}"

        prompts.append(prompt)

    # 生成最终的 prompt
    final_prompt = [first_instruction] if first_instruction else []
    final_prompt.append(prefix)
    final_prompt.extend(prompts)

    return '\n'.join(final_prompt)


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
        data_path = os.path.join(os.getcwd(), "data/ICL_Examples")

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
    # current_data_path = os.path.join(os.path.dirname(__file__), "../../data/ICL_Examples")
    # print(prepare_examples(data_path=current_data_path, task='da', language='ch'))
    # print(prepare_examples(data_path=current_data_path, task='ee', language='en'))
    # print(prepare_examples(data_path=current_data_path, task='ner', language='ch'))
    # print(prepare_examples(data_path=current_data_path, task='re', language='en'))
    # print(prepare_examples(data_path=current_data_path, task='rte', language='ch'))

    # file_path = os.path.join(os.path.dirname(__file__), "../../data/RE/test.json")
    # print(prepare_input_plus(file_path, language='en'))

    pass
