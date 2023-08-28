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
    2: ['"实体类型：实体\n"', entity_convert_target2],
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


# 2.常见文本主题及其schema

根据统计分析, 我们将文本划分为以下12种主题：

1. 人物（人、虚拟的人、名称等）
2. 地理地区（城市、国家、河流、山脉、大洲、湖泊等）
3. 事件（战争、活动、赛事等）
4. 组织（企业、政府机构、机构、家族、队伍等）
5. 生物（动物、植物、微生物、种等）
6. 人造物件（材料、食物、设备等）
7. 自然科学（化合物、数学等）
8. 医学（健康问题、生理状况等）
9. 运输（物流、航空、铁路系统、船运等）
10. 建筑（建筑物、设施等）
11. 天文对象（天体等）
12. 作品（电影、数据、音乐等）

并且在 [schema](./kg2instruction/schema.py) 中我们提供了每个主题下常见的关系类型。

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


# 3. 转换脚本

提供一个名为 [convert.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert.py)、[convert_test.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert_test.py) 的脚本，用于将数据统一转换为可以直接输入 KnowLM 的指令。在执行 convert.py 之前，请参考 [data](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/data) 目录中包含了每个任务的预期数据格式。

```bash
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \       # 不同语言使用的template及转换脚本不同
  --task NER \          # ['RE', 'NER', 'EE']三种任务
  --sample 0 \          # 若为-1, 则从4种指令和4种输出格式中随机采样其中一种, 否则即为指定的指令格式, -1<=sample<=3
  --all                 # 是否将指令中指定的抽取类型列表设置为全部schema
```

[convert_test.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert_test.py) 不要求数据具有标签(`entity`、`relation`、`event`)字段, 只需要具有 `input` 字段, 以及提供 `schema_path`, 适合用来处理测试数据。

```bash
python kg2instruction/convert_test.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/processed.json \
    --schema_path data/NER/schema.json \
    --language zh \      
    --task NER \          
    --sample 0 
```


# 4.现成数据集

下面是一些现成的处理后的数据：

| 名称                  | 下载                                                                                                                     | 数量     | 描述                                                                                                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KnowLM-IE.json       | [Google drive](https://drive.google.com/file/d/1hY_R6aFgW4Ga7zo41VpOVOShbTgBqBbL/view?usp=sharing) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)      | 281860 | [InstructIE](https://arxiv.org/abs/2305.11527) 中提到的数据集                                                                                     |
| KnowLM-ke         | [HuggingFace](hhttps://huggingface.co/datasets/zjunlp/knowlm-ke)                     | XXXX   | 训练[zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi)所用到的所有指令数据(通用、IE、Code、COT等) |


`KnowLM-IE.json`：包含 `'id'`(唯一标识符)、`'cate'`(文本主题)、`'instruction'`(抽取指令)、`'input'`(输入文本)、`'output'`(输出文本)字段、`'relation'`(三元组)字段，可以通过`'relation'`自由构建抽取的指令和输出，`'instruction'`有16种格式(4种prompt * 4种输出格式)，`'output'`是按照`'instruction'`中指定的输出格式生成的文本。


`KnowLM-ke`：仅包含`'instruction'`、`'input'`、`'output'`字段。其目录下的`ee-en.json`、`ee_train.json`、`ner-en.json`、`ner_train.json`、`re-en.json`、`re_train.json`为中英文IE指令数据。



# 5. 评估
我们提供一个位于 [evaluate.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/evaluate.py) 的脚本，用于将模型的字符串输出转换为列表并计算 F1 分数。

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task ner \
  --language zh
```

