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
    # print('---------------------------------')

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


# test
if __name__ == "__main__":
    print(prepare_examples(data_path='../data', task='da', language='ch'))
    print(prepare_examples(data_path='../data', task='ee', language='ch'))
    print(prepare_examples(data_path='../data', task='ner', language='ch'))
    print(prepare_examples(data_path='../data', task='re', language='ch'))
    print(prepare_examples(data_path='../data', task='rte', language='ch'))



"""
[{'context': '孔正锡，导演，2005年以一部温馨的爱情电影《长腿叔叔》敲开电影界大门。', 'head_type': '电影名称', 'head_entity': '长腿叔叔', 'tail_type': '人名', 'tail_entity': '孔正锡', 'relation': '导演'}, {'context': '《我家有喜2》（又名《开门见喜》）是由丁仰国导演执导，张佳编剧，海陆、高梓淇、万茜、李佳航、关雪盈、陈晓联袂主演的都市轻喜剧。', 'head_type': '电视剧名称', 'head_entity': '我家有喜', 'tail_type': '人名', 'tail_entity': '丁仰国', 'relation': '导演'}]
=============================================
[{'input': '历经4小时51分钟的体力、意志力鏖战，北京时间9月9日上午纳达尔在亚瑟·阿什球场，以7比5、6比3、5比7、4比6和6比4击败赛会5号种子俄罗斯球员梅德韦杰夫，夺得了2019年美国网球公开赛男单冠军。', 'output': [{'event_type': '竞赛行为-胜负', 'arguments': [{'role': '时间', 'argument': '北京时间9月9日上午'}, {'role': '胜者', 'argument': '纳达尔'}, {'role': '败者', 'argument': '5号种子俄罗斯球员梅德韦杰夫'}, {'role': '赛事名称', 'argument': '2019年美国网球公开赛'}]}, {'event_type': '竞赛行为-夺冠', 'arguments': [{'role': '时间', 'argument': '北京时间9月9日上午'}, {'role': '夺冠赛事', 'argument': '2019年美国网球公开赛'}, {'role': '冠军', 'argument': '纳达尔'}]}]}]
=============================================
[{'output': [{'E': 'ORG', 'W': '国务院'}, {'E': 'PER', 'W': '秦始皇'}, {'E': 'LOC', 'W': '陕西省'}, {'E': 'LOC', 'W': '西安市'}, {'E': 'TIME', 'W': '1961年'}]}]
============================================= RE（Relation Extraction，关系抽取）
[{'context': '孔正锡，导演，2005年以一部温馨的爱情电影《长腿叔叔》敲开电影界大门。', 'head_type': '电影名称', 'head_entity': '长腿叔叔', 'tail_type': '人名', 'tail_entity': '孔正锡', 'relation': '导演'}, {'context': '《我家有喜2》（又名《开门见喜》）是由丁仰国导演执导，张佳编剧，海陆、高梓淇、万茜、李佳航、关雪盈、陈晓联袂主演的都市轻喜剧。', 'head_type': '电视剧名称', 'head_entity': '我家有喜', 'tail_type': '人名', 'tail_entity': '丁仰国', 'relation': '导演'}]
============================================= RTE（Recognizing Textual Entailment，文本蕴涵识别）
[{'input': '《来自星星的你》是由张太侑导演，朴智恩编剧，于2013年12月18日在韩国SBS电视台首播', 'output': [['《来自星星的你》', '导演', '张太侑'], ['《来自星星的你》', '编剧', '朴智恩'], ['《来自星星的你》', '播出', '韩国SBS电视台']]}]



da：
{
  "context": "孔正锡，导演，2005年以一部温馨的爱情电影《长腿叔叔》敲开电影界大门。",
  "head_type": "电影名称",
  "head_entity": "长腿叔叔",
  "tail_type": "人名",
  "tail_entity": "孔正锡",
  "relation": "导演"
}

ee:
{
  "input": "历经4小时51分钟的体力、意志力鏖战，北京时间9月9日上午纳达尔在亚瑟·阿什球场，以7比5、6比3、5比7、4比6和6比4击败赛会5号种子俄罗斯球员梅德韦杰夫，夺得了2019年美国网球公开赛男单冠军。",
  "output": [
    {
      "event_type": "竞赛行为-胜负",
      "arguments": [
        {"role": "时间", "argument": "北京时间9月9日上午"},
        {"role": "胜者", "argument": "纳达尔"},
        {"role": "败者", "argument": "5号种子俄罗斯球员梅德韦杰夫"},
        {"role": "赛事名称", "argument": "2019年美国网球公开赛"}
      ]
    }
  ]
}

ner:
{
  "output": [
    {"E": "ORG", "W": "国务院"},
    {"E": "PER", "W": "秦始皇"},
    {"E": "LOC", "W": "陕西省"},
    {"E": "LOC", "W": "西安市"},
    {"E": "TIME", "W": "1961年"}
  ]
}

re:
{
  "context": "孔正锡，导演，2005年以一部温馨的爱情电影《长腿叔叔》敲开电影界大门。",
  "head_type": "电影名称",
  "head_entity": "长腿叔叔",
  "tail_type": "人名",
  "tail_entity": "孔正锡",
  "relation": "导演"
}

rte:
{
  'input': '《来自星星的你》是由张太侑导演，朴智恩编剧，于2013年12月18日在韩国SBS电视台首播',
  'output': [
    ['《来自星星的你》', '导演', '张太侑'],
    ['《来自星星的你》', '编剧', '朴智恩'],
    ['《来自星星的你》', '播出', '韩国SBS电视台']
  ]
}
"""
