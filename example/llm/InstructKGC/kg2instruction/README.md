

- [1.IE template](#1ie-template)
- [2. Common text topics and their schemas](#2-common-text-topics-and-their-schemas)
- [3.Convert script](#3convert-script)
- [4.Datasets](#4datasets)
- [5.Evaluate](#5evaluate)

# 1.IE template
NER supports the following templates:
```python
entity_template_zh =  {
    0:'已知候选的实体类型列表：{s_schema}，请你根据实体类型列表，从以下输入中抽取出可能存在的实体。请按照{s_format}的格式回答。',
    1:'我将给你个输入，请根据实体类型列表：{s_schema}，从输入中抽取出可能包含的实体，并以{s_format}的形式回答。',
    2:'我希望你根据实体类型列表从给定的输入中抽取可能的实体，并以{s_format}的格式回答，实体类型列表={s_schema}。',
    3:'给定的实体类型列表是{s_schema}\n根据实体类型列表抽取，在这个句子中可能包含哪些实体？你可以先别出实体, 再判断实体类型。请以{s_format}的格式回答。',
}

entity_int_out_format_zh = {
    0:['"(实体,实体类型)"', entity_convert_target0],
    1:['"实体是\n实体类型是\n\n"', entity_convert_target1],
    2:['"实体类型：实体\n"', entity_convert_target2],
    3:["JSON字符串[{'entity':'', 'entity_type':''}, ]", entity_convert_target3],
}

entity_template_en =  {
    0:'Identify the entities and types in the following text and where entity type list {s_schema}. Please provide your answerin the form of {s_format}.',
    1:'From the given text, extract the possible entities and types . The types are {s_schema}. Please format your answerin the form of {s_format}.', 
}

entity_int_out_format_en = {
    0:['(Entity, Type)', entity_convert_target0_en],
    1:["{'Entity':'', 'Type':''}", entity_convert_target1_en],
}
```

Both the schema and format placeholders ({s_schema} and {s_format}) are embedded within the templates and must be specified by users.

For a more comprehensive understanding of the templates, please refer to the files [ner_template.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/ner_template.py)、[re_template.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/re_template.py)、[ee_template.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/ee_template.py) .


# 2.Common text topics and their schemas

According to statistical analysis, we have categorized text into the following 12 topics:

1. Characters (individuals, fictional characters, names, etc.)
2. Geographic Locations (cities, countries, rivers, mountains, continents, lakes, etc.)
3. Events (wars, activities, competitions, etc.)
4. Organizations (businesses, government institutions, organizations, families, teams, etc.)
5. Biological Entities (animals, plants, microorganisms, species, etc.)
6. Artifacts (materials, food, equipment, etc.)
7. Natural Sciences (compounds, mathematics, etc.)
8. Medical (health issues, physiological conditions, etc.)
9. Transportation (logistics, aviation, railway systems, shipping, etc.)
10. Structures (buildings, facilities, etc.)
11. Astronomical Objects (celestial bodies, etc.)
12. Works (movies, data, music, etc.)

Moreover, in the [schema](./kg2instruction/schema.py) provided, we have listed common relationship types under each topic.

```python
{
    '组织': ['别名', '位于', '类型', '成立时间', '解散时间', '成员', '创始人', '事件', '子组织', '产品', '成就', '运营'], 
    '医学': ['别名', '病因', '症状', '可能后果', '包含', '发病部位'], 
    '事件': ['别名', '类型', '发生时间', '发生地点', '参与者', '主办方', '提名者', '获奖者', '赞助者', '获奖作品', '获胜者', '奖项'], 
    '运输': ['别名', '位于', '类型', '属于', '途径', '开通时间', '创建时间', '车站等级', '长度', '面积'], 
    '人造物件': ['别名', '类型', '受众', '成就', '品牌', '产地', '长度', '宽度', '高度', '重量', '价值', '制造商', '型号', '生产时间', '材料', '用途', '发现者或发明者'], 
    '生物': ['别名', '学名', '类型', '分布', '父级分类单元', '主要食物来源', '用途', '长度', '宽度', '高度', '重量', '特征'], 
    '建筑': ['别名', '类型', '位于', '临近', '名称由来', '长度', '宽度', '高度', '面积', '创建时间', '创建者', '成就', '事件'], 
    '自然科学': ['别名', '类型', '性质', '生成物', '用途', '组成', '产地', '发现者或发明者'], 
    '地理地区': ['别名', '类型', '所在行政领土', '接壤', '事件', '面积', '人口', '行政中心', '产业', '气候'], 
    '作品': ['别名', '类型', '受众', '产地', '成就', '导演', '编剧', '演员', '平台', '制作者', '改编自', '包含', '票房', '角色', '作曲者', '作词者', '表演者', '出版时间', '出版商', '作者'], 
    '人物': ['别名', '籍贯', '国籍', '民族', '朝代', '出生时间', '出生地点', '死亡时间', '死亡地点', '专业', '学历', '作品', '职业', '职务', '成就', '所属组织', '父母', '配偶', '兄弟姊妹', '亲属', '同事', '参与'], 
    '天文对象': ['别名', '类型', '坐标', '发现者', '发现时间', '名称由来', '属于', '直径', '质量', '公转周期', '绝对星等', '临近']
}
```


# 3.Convert script

A script named [convert.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert.py)、[convert_test.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert_test.py) is provided to facilitate the uniform conversion of data into KnowLM instructions. The [data](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/data) directory contains the expected data format for each task before executing convert.py.

```bash
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \
  --task NER \
  --sample 0 \
  --all
```

[convert_test.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert_test.py) does not require data to have label (`entity`, `relation`, `event`) fields, only needs to have an `input` field and provide a `schema_path` is suitable for processing test data.

```bash
python kg2instruction/convert_test.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/processed.json \
    --schema_path data/NER/schema.json \
    --language zh \      
    --task NER \          
    --sample 0 
```


# 4.Datasets


Below are some readily processed datasets:

| Name                | Download Links                                                                                                           | Quantity | Description                                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KnowLM-IE.json       | [Google Drive](https://drive.google.com/file/d/1hY_R6aFgW4Ga7zo41VpOVOShbTgBqBbL/view?usp=sharing) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)      | 281860 | Dataset mentioned in [InstructIE](https://arxiv.org/abs/2305.11527)                                                                                 |
| KnowLM-ke         | [HuggingFace](hhttps://huggingface.co/datasets/zjunlp/knowlm-ke)                     | XXXX   | Contains all instruction data (General, IE, Code, COT, etc.) used for training [zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi) |


`KnowLM-IE.json`: Contains fields such as `'id'` (unique identifier), `'cate'` (text category), `'instruction'` (extraction instruction), `'input'` (input text), `'output'` (output text), and `'relation'` (triples). The `'relation'` field can be used to construct extraction instructions and outputs freely. `'instruction'` has 16 formats (4 prompts * 4 output formats), and `'output'` is generated in the specified format from `'instruction'`.

`KnowLM-ke`: Contains fields `'instruction'`, `'input'`, and `'output'` only. The files `ee-en.json`, `ee_train.json`, `ner-en.json`, `ner_train.json`, `re-en.json`, and `re_train.json` under its directory contain Chinese-English IE instruction data.




# 5.Evaluate

We provide a script at [evaluate.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/evaluate.py) to convert the string output of the model into a list and calculate F1

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task ner \
  --language zh
```


