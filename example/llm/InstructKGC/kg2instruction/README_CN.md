- [1.信息抽取模板](#1信息抽取模板)
- [2.常见文本主题及其schema](#2常见文本主题及其schema)
- [3.转换脚本](#3转换脚本)
- [4.现成数据集](#4现成数据集)
- [5.评估](#5评估)


# 1.信息抽取模板
命名实体识别（NER）支持以下输出格式, prompt模版请参考[configs](../configs)：

```python
entity_int_out_format_zh = {
    0: ['"(实体,实体类型)"', entity_convert_target0],
    1: ['"实体是\n实体类型是\n\n"', entity_convert_target1],
    2: ['"实体类型：实体\n"', entity_convert_target2],
    3: ["JSON字符串[{'entity':'', 'entity_type':''}, ]", entity_convert_target3],
}

entity_int_out_format_en = {
    0:['(Entity,Entity Type)\n', entity_convert_target0],
    1:['Entity is,Entity Type is\n', entity_convert_target1_en],
    2:['Entity Type：Entity\n', entity_convert_target2],
    3:["{'entity':'', 'entity_type':''}\n", entity_convert_target3],
} 
```


这些模板中的schema（{s_schema}）和输出格式 （{s_format}）占位符被嵌入在模板中，用户必须指定。
有关模板的更全面理解，请参阅文件  [ner_converter.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert/converter/ner_converter.py)、[re_converter.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert/converter/re_converter.py)、[ee_converter.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert/converter/ee_converter.py) .


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
wiki_cate_schema_zh = {
    '人物': ['出生地点', '出生日期', '国籍', '职业', '作品', '成就', '籍贯', '职务', '配偶', '父母', '别名', '所属组织', '死亡日期', '兄弟姊妹', '墓地'], 
    '地理地区': ['位于', '别名', '人口', '行政中心', '面积', '成就', '长度', '宽度', '海拔'], 
    '建筑': ['位于', '别名', '成就', '事件', '创建时间', '宽度', '长度', '创建者', '高度', '面积', '名称由来'], 
    '作品': ['作者', '出版时间', '别名', '产地', '改编自', '演员', '出版商', '成就', '表演者', '导演', '制片人', '编剧', '曲目', '作曲者', '作词者', '制作商', '票房', '出版平台'], 
    '生物': ['分布', '父级分类单元', '长度', '主要食物来源', '别名', '学名', '重量', '宽度', '高度'], 
    '人造物件': ['别名', '品牌', '生产时间', '材料', '产地', '用途', '制造商', '发现者或发明者'], 
    '自然科学': ['别名', '性质', '组成', '生成物', '用途', '产地', '发现者或发明者'], 
    '组织': ['位于', '别名', '子组织', '成立时间', '产品', '成就', '成员', '创始人', '解散时间', '事件'], 
    '运输': ['位于', '创建时间', '线路', '开通时间', '途经', '面积', '别名', '长度', '宽度', '成就', '车站等级'], 
    '事件': ['参与者', '发生地点', '发生时间', '别名', '赞助者', '伤亡人数', '起因', '导致', '主办方', '所获奖项', '获胜者'], 
    '天文对象': ['别名', '属于', '发现或发明时间', '发现者或发明者', '名称由来', '绝对星等', '直径', '质量'], 
    '医学': ['症状', '别名', '发病部位', '可能后果', '病因']
}
```


# 3.转换脚本

提供一个名为 [convert.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert.py)、[convert_test.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert_test.py) 的脚本，用于将数据统一转换为可以直接输入 KnowLM 的指令。在执行 convert.py 之前，请参考 [data](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/data) 目录中包含了每个任务的预期数据格式。

```bash
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \      # 不同语言使用的template及转换脚本不同
  --task NER \         # ['RE', 'NER', 'EE', 'EET', 'EEA'] 5种任务
  --sample -1 \        # 若为-1, 则从20种指令和4种输出格式中随机采样其中一种, 否则即为指定的指令格式, -1<=sample<20
  --neg_ratio 1 \      # 表示所有样本的负采样比例
  --neg_schema 1 \     # 表示从schema中负采样的比例
  --random_sort        # 是否对指令中的schema列表进行随机排序
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
| InstructIE-train       | [Google drive](https://drive.google.com/file/d/1VX5buWC9qVeVuudh_mhc_nC7IPPpGchQ/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [百度云盘](https://pan.baidu.com/s/1xXVrjkinw4cyKKFBR8BwQw?pwd=x4s7)    | 30w+ | InstructIE训练集                                                                                     |
| InstructIE-valid       | [Google drive](https://drive.google.com/file/d/1EMvqYnnniKCGEYMLoENE1VD6DrcQ1Hhj/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [百度云盘](https://pan.baidu.com/s/11u_f_JT30W6B5xmUPC3enw?pwd=71ie)    | 2000+ | InstructIE验证集                                                                                     |
| InstructIE-test       | [Google drive](https://drive.google.com/file/d/1WdG6_ouS-dBjWUXLuROx03hP-1_QY5n4/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [百度云盘](https://pan.baidu.com/s/1JiRiOoyBVOold58zY482TA?pwd=cyr9)     | 2000+ | InstructIE测试集                                                                                     |
| train.json, valid.json          | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing)                     | 5000   | [CCKS2023 开放环境下的知识图谱构建与补全评测任务一：指令驱动的自适应知识图谱构建](https://tianchi.aliyun.com/competition/entrance/532080/introduction) 中的初赛训练集及测试集 |


`InstructIE-train`包含`InstructIE-zh.json`、`InstructIE-en.json`两个文件, 每个文件均包含以下字段：`'id'`(唯一标识符)、`'cate'`(文本主题)、`'entity'`、`'relation'`(三元组)字段，可以通过`'entity'`、`'relation'`自由构建抽取的指令和输出。
`InstructIE-valid`、`InstructIE-test`分别是验证集和测试集, 包含`zh`和`en`双语。

`train.json`：字段含义同`train.json`，`'instruction'`、`'output'`都只有1种格式，也可以通过`'relation'`自由构建抽取的指令和输出。
`valid.json`：字段含义同`train.json`，但是经过众包标注，更加准确。


以下是各字段的说明：

|    字段     |                          说明                          |
| :---------: | :----------------------------------------------------: |
|     id      |                     唯一标识符                           |
|    cate     |     文本input对应的主题(共12种)                           |
|    input    |    模型输入文本（需要抽取其中涉及的所有关系三元组）            |
| instruction |                 模型进行抽取任务的指令                     |
| output      |                   模型期望输出                           |
| entity      |            实体(entity, entity_type)                    |
| relation    |     input中涉及的关系三元组(head, relation, tail)         |




# 5.评估
我们提供一个位于 [evaluate.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/evaluate.py) 的脚本，用于将模型的字符串输出转换为列表并计算 F1 分数。

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task NER \
  --language zh
```

