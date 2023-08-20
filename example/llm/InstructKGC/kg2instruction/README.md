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
    2:['"实体：实体类型\n"', entity_convert_target2],
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

# 2.Convert script

A script named [convert.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert.py) is provided to facilitate the uniform conversion of data into KnowLM instructions. The [data](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/data) directory contains the expected data format for each task before executing convert.py.


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

# 3.Evaluate

We provide a script at [evaluate.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/evaluate.py) to convert the string output of the model into a list and calculate F1

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task ner \
  --language zh
```


