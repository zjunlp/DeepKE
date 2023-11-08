

- [1.IE template](#1ie-template)
- [2.Common text topics and their schemas](#2common-text-topics-and-their-schemas)
- [3.Convert script](#3convert-script)
- [4.Datasets](#4datasets)
- [5.Evaluate](#5evaluate)

# 1.IE template

Named Entity Recognition (NER) supports the following output formats, please refer to the [configs](../configs) for prompt templates:

```python
entity_int_out_format_zh = {
    0:['"(实体,实体类型)"', entity_convert_target0],
    1:['"实体是\n实体类型是\n\n"', entity_convert_target1],
    2:['"实体类型：实体\n"', entity_convert_target2],
    3:["JSON字符串[{'entity':'', 'entity_type':''}, ]", entity_convert_target3],
}

self.entity_int_out_format_en = {
    0:['(Entity,Entity Type)\n', self.entity_convert_target0],
    1:['Entity is,Entity Type is\n', self.entity_convert_target1_en],
    2:['Entity Type：Entity\n', self.entity_convert_target2],
    3:["{'entity':'', 'entity_type':''}\n", self.entity_convert_target3],
} 
```

Both the schema and format placeholders ({s_schema} and {s_format}) are embedded within the templates and must be specified by users.

For a more comprehensive understanding of the templates, please refer to the files [ner_converter.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert/converter/ner_converter.py)、[re_converter.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert/converter/re_converter.py)、[ee_converter.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert/converter/ee_converter.py)  .


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
wiki_cate_schema_en =  {
    'Person': ['place of birth', 'date of birth', 'country of citizenship', 'occupation', 'work', 'achievement', 'ancestral home', 'position', 'spouse', 'parent', 'alternative name', 'affiliated organization', 'date of death', 'sibling', 'place of death'], 
    'Geographic_Location': ['located in', 'alternative name', 'population', 'capital', 'area', 'achievement', 'length', 'width', 'elevation'], 
    'Building': ['located in', 'alternative name', 'achievement', 'event', 'creation time', 'width', 'length', 'creator', 'height', 'area', 'named after'], 
    'Works': ['author', 'publication date', 'alternative name', 'country of origin', 'based on', 'cast member', 'publisher', 'achievement', 'performer', 'director', 'producer', 'screenwriter', 'tracklist', 'composer', 'lyricist', 'production company', 'box office', 'publishing platform'], 
    'Creature': ['distribution', 'parent taxon', 'length', 'main food source', 'alternative name', 'taxon name', 'weight', 'width', 'height'], 
    'Artificial_Object': ['alternative name', 'brand', 'production date', 'made from material', 'country of origin', 'has use', 'manufacturer', 'discoverer or inventor'], 
    'Natural_Science': ['alternative name', 'properties', 'composition', 'product', 'has use', 'country of origin', 'discoverer or inventor', 'causes'], 
    'Organization': ['located in', 'alternative name', 'has subsidiary', 'date of incorporation', 'product', 'achievement', 'member', 'founded by', 'dissolution time', 'event'], 
    'Transport': ['located in', 'inception', 'connecting line', 'date of official opening', 'pass', 'area', 'alternative name', 'length', 'width', 'achievement', 'class of station'], 
    'Event': ['participant', 'scene', 'occurrence time', 'alternative name', 'sponsor', 'casualties', 'has cause', 'has effect', 'organizer', 'award received', 'winner'], 
    'Astronomy': ['alternative name', 'of', 'time of discovery or invention', 'discoverer or inventor', 'name after', 'absolute magnitude', 'diameter', 'mass'], 
    'Medicine': ['symptoms', 'alternative name', 'affected body part', 'possible consequences', 'etiology']
}
```


# 3.Convert script

A script named [convert.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert.py)、[convert_test.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/convert_test.py) is provided to facilitate the uniform conversion of data into KnowLM instructions. The [data](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/data) directory contains the expected data format for each task before executing convert.py.

```bash
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \      # Different templates and conversion scripts are used for different languages
  --task NER \         # 5 types of tasks: ['RE', 'NER', 'EE', 'EET', 'EEA']
  --sample -1 \        # If -1, randomly sample one from 20 instruction types and 4 output formats, otherwise it is the specified instruction format, -1<=sample<20
  --neg_ratio 1 \      # Indicates the negative sampling ratio for all samples
  --neg_schema 1 \     # Indicates the negative sampling ratio from the schema
  --random_sort        # Whether to randomly sort the schema list in the instructions

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

| Name                   | Download                                                     | Quantity | Description                                                  |
| ---------------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| InstructIE-train          | [Google drive](https://drive.google.com/file/d/1VX5buWC9qVeVuudh_mhc_nC7IPPpGchQ/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/1xXVrjkinw4cyKKFBR8BwQw?pwd=x4s7)  | 30w+  | InstructIE train set |
| InstructIE-valid       | [Google drive](https://drive.google.com/file/d/1EMvqYnnniKCGEYMLoENE1VD6DrcQ1Hhj/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/11u_f_JT30W6B5xmUPC3enw?pwd=71ie)     | 2000+ | InstructIE validation set                                                                                     |
| InstructIE-test       | [Google drive](https://drive.google.com/file/d/1WdG6_ouS-dBjWUXLuROx03hP-1_QY5n4/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)  <br/> [Baidu Netdisk](https://pan.baidu.com/s/1JiRiOoyBVOold58zY482TA?pwd=cyr9)     | 2000+ | InstructIE test set                                                                                    |
| train.json, valid.json | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing) | 5,000    | Preliminary training set and test set for the task "Instruction-Driven Adaptive Knowledge Graph Construction" in [CCKS2023 Open Knowledge Graph Challenge](https://tianchi.aliyun.com/competition/entrance/532080/introduction), randomly selected from instruct_train.json |


`InstrumentIE-train` contains two files: `InstrumentIE-zh.json` and `InstrumentIE-en.json`, each of which contains the following fields: `'id'` (unique identifier), `'cate'` (text category), `'entity'` and `'relation'` (triples) fields. The extracted instructions and output can be freely constructed through `'entity'` and `'relation'`.

`InstrumentIE-valid` and `InstrumentIE-test` are validation sets and test sets, respectively, including bilingual `zh` and `en`.

`train.json`: Same fields as `KnowLM-IE.json`, `'instruction'` and `'output'` have only one format, and extraction instructions and outputs can also be freely constructed through `'relation'`.

`valid.json`: Same fields as `train.json`, but with more accurate annotations achieved through crowdsour

Here is an explanation of each field:

|    Field    |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|     id      |                   Unique identifier                   |
|    cate     |     text topic of input (12 topics in total)                        |
|    input    | Model input text (need to extract all triples involved within) |
| instruction |   Instruction for the model to perform the extraction task   |
|    output   | Expected model output |
| entity      |            entities(entity, entity_type)                    |
|   relation  |             Relation triples(head, relation, tail) involved in the input             |



# 5.Evaluate

We provide a script at [evaluate.py](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/kg2instruction/evaluate.py) to convert the string output of the model into a list and calculate F1

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task NER \
  --language zh
```


