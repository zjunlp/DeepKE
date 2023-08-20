# 1. 信息抽取模板
命名实体识别（NER）支持以下模板：

```python
entity_template_zh = {
    0: '已知候选的实体类型列表：{s_schema}，请你根据实体类型列表，从以下输入中抽取出可能存在的实体。请按照{s_format}的格式回答。',
    1: '我将给你个输入，请根据实体类型列表：{s_schema}，从输入中抽取出可能包含的实体，并以{s_format}的形式回答。',
    2: '我希望你根据实体类型列表从给定的输入中抽取可能的实体，并以{s_format}的格式回答，实体类型列表={s_schema}。',
    3: '给定的实体类型列表是{s_schema}\n根据实体类型列表抽取，在这个句子中可能包含哪些实体？你可以先别出实体，再判断实体类型。请以{s_format}的格式回答。',
}

entity_int_out_format_zh = {
    0: ['"(实体,实体类型)"', entity_convert_target0],
    1: ['"实体是\n实体类型是\n\n"', entity_convert_target1],
    2: ['"实体：实体类型\n"', entity_convert_target2],
    3: ["JSON字符串[{'entity':'', 'entity_type':''}, ]", entity_convert_target3],
}

entity_template_en = {
    0: 'Identify the entities and types in the following text and where entity type list {s_schema}. Please provide your answer in the form of {s_format}.',
    1: 'From the given text, extract the possible entities and types. The types are {s_schema}. Please format your answer in the form of {s_format}.',
}

entity_int_out_format_en = {
    0: ['(Entity, Type)', entity_convert_target0_en],
    1: ["{'Entity':'', 'Type':''}", entity_convert_target1_en],
}
```


这些模板中的schema（{s_schema}）和输出格式 （{s_format}）占位符被嵌入在模板中，用户必须指定。
有关模板的更全面理解，请参阅文件  [ner_template.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/ner_template.py)、[re_template.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/re_template.py)、[ee_template.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/ee_template.py) .



# 2. 转换脚本

提供一个名为 [convert.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert.py) 的脚本，用于将数据统一转换为可以直接输入 KnowLM 的指令。在执行 convert.py 之前，请参考 [data](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/data) 目录中包含了每个任务的预期数据格式。

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

# 3. 评估
我们提供一个位于 [evaluate.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/evaluate.py) 的脚本，用于将模型的字符串输出转换为列表并计算 F1 分数。

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task ner \
  --language zh
```

