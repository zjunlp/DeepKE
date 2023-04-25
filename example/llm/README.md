<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md">简体中文</a> </b>
</p>

## Contents

- [Contents](#Contents)
- [IE with Large Language Models](#IE-with-Large-Language-Models)
  - [Requirement, Data and Configuration](#Requirement,-Data-and-Configuration)
  - [Run and Examples](#Run-and-Examples)
- [Data Augmentation with Large Language Models](#Data-Augmentation-with-Large-Language-Models)
  - [Configuration](#Configuration)
- [InstructionKGC (CCKS2023)](#InstructKGC-CCKS2023-Evaluation-of-Instruction-based-Knowledge-Graph-Construction)
- [CodeKGC](#CodeKGC-Code-Language-Models-for-Knowledge-Graph-Construction)

# IE with Large Language Models

## Requirement, Data and Configuration

- Requirement

  The LLM module of Deepke calls the [EasyInstruct](https://github.com/zjunlp/EasyInstruct) tookit.

  ```
  >> pip install easyinstruct
  >> pip install hydra-core
  ```

- Data

  The data here refers to the examples data used for in-context learning, which is stored in the `data` folder. The `.json` files in it are the default examples data for various tasks. Users can customize the examples in them, but they need to follow the given data format.

- Configuration

  The `conf` folder stores the set parameters. The parameters required to call the GPT3 interface are passed in through the files in this folder.

  - In the Named Entity Recognition (ner) task, `text_input` parameter is the prediction text; `domain` is the domain of the prediction text, which can be empty; `label` is the entity label set, which can also be empty. 

  - In the Relation Extraction (re) task, `text_input` parameter is the text; `domain` indicates the domain to which the text belongs, and it can be empty; `labels` is the set of relationship type labels. If there is no custom label set, this parameter can be empty; `head_entity` and `tail_entity` are the head entity and tail entity of the relationship to be predicted, respectively; `head_type` and `tail_type` are the types of the head and tail entities to be predicted in the relationship.

  - In the Event Extraction (ee) task, `text_input` parameter is the prediction text; `domain` is the domain of the prediction text, which can also be empty. 

  - In the Relational Triple Extraction (rte) task, `text_input` parameter is the prediction text; `domain` is the domain of the prediction text, which can also be empty.

  - The specific meanings of other parameters are as follows:
    - `task` parameter is used to specify the task type, where `ner` represents named entity recognition task, `re` represents relation extraction task, `ee` represents event extraction task, and `rte` represents triple extraction task;
    - `language` indicates the language of the task, where `en` represents English extraction tasks, and `ch` represents Chinese extraction tasks;
    - `engine` indicates the name of the large model used, which should be consistent with the model name specified by the OpenAI API;
    - `api_key` is the user's API key;
    - `zero_shot` indicates whether zero-shot setting is used. When it is set to `True`, only the instruction prompt model is used for information extraction, and when it is set to `False`, in-context form is used for information extraction;
    - `instruction` parameter is used to specify the user-defined prompt instruction, and the default instruction is used when it is empty;
    - `data_path` indicates the directory where in-context examples are stored, and the default is the `data` folder.


## Run and Examples

Once the parameters are set, you can directly run the `run.py`：

```
>> python run.py
```

Below are input and output examples for different tasks:

| Task |                            Input                             |                            Output                            |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| NER  | Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday. | [{'E': 'Country', 'W': 'Japan'}, {'E': 'Country', 'W': 'Syria'}, {'E': 'Competition', 'W': 'Asian Cup'}, {'E': 'Competition', 'W': 'Group C championship'}] |
|  RE  | The Dutch newspaper Brabants Dagblad said the boy was probably from Tilburg in the southern Netherlands and that he had been on safari in South Africa with his mother Trudy , 41, father Patrick, 40, and brother Enzo, 11. |                           parents                            |
|  EE  | In Baghdad, a cameraman died when an American tank fired on the Palestine Hotel. | event_list: [ event_type: [arguments: [role: "cameraman", argument: "Baghdad"], [role: "American tank", argument: "Palestine Hotel"]] ] |
| RTE  |    The most common audits were about waste and recycling.    | [['audit', 'type', 'waste'], ['audit', 'type', 'recycling']] |

# Data Augmentation with Large Language Models

To compensate for the lack of labeled data in few-shot scenarios for relation extraction, we have designed prompts with data style descriptions to guide large language models to automatically generate more labeled data based on existing few-shot data.

## Configuration

- Set `task` to `da`;
- Set `text_input` to the relationship label to be enhanced, such as `org:founded_by`;
- Set `zero_shot` to `False` and set the low-sample example in the corresponding file under the `data` folder for the `da` task;
- The range of entity labels can be specified in `labels`.

Here is an example of a data augmentation `prompt`:

```python
'''
One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. 

The head entity has the relation with the tail entity and entities are pre-categorized as the following types: URL, LOCATION, IDEOLOGY, CRIMINAL CHARGE, TITLE, STATE OR PROVINCE, DATE, PERSON, NUMBER, CITY, DURATION, CAUSE OF DEATH, COUNTRY, NATIONALITY, RELIGION, ORGANIZATION, MISCELLANEOUS. 

Here are some samples for relation 'org:founded_by':

Relation: org:founded_by. Context: Talansky is also the US contact for the New Jerusalem Foundation , an organization founded by Olmert while he was Jerusalem 's mayor . Head Entity: New Jerusalem Foundation. Head Type: ORGANIZATION. Tail Entity: Olmert. Tail Type: PERSON.

Relation: org:founded_by. Context: Sharpton has said he will not endorse any candidate until hearing more about their views on civil rights and other issues at his National Action Network convention next week in New York City . Head Entity: National Action Network. Head Type: ORGANIZATION. Tail Entity: his. Tail Type: PERSON.

Relation: org:founded_by. Context: `` We believe that we can best serve our clients by offering a single multistrategy hedge fund platform , '' wrote John Havens , who was a founder of Old Lane with Pandit and is president of the alternative investment group . Head Entity: Old Lane. Head Type: ORGANIZATION. Tail Entity: John Havens. Tail Type: PERSON.

Generate more samples for the relation 'org:founded_by'.
'''
```

# InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction

The following is a baseline description of the *ChatGPT/GPT-4* for the **Instruction-based Knowledge Graph Construction** task in the **[CCKS2023 Open Environment Knowledge Graph Construction and Completion Evaluation competition](https://tianchi.aliyun.com/competition/entrance/532080/introduction?spm=5176.12281957.0.0.4c885d9b2YX9Nu)**.

## Task Object

Extract relevant entities and relations according to user input instructions to construct a knowledge graph. This task may include knowledge graph completion, where the model is required to complete missing triples while extracting entity-relation triples.

Below is an example of a **Knowledge Graph Construction Task**. Given an input text `text` and an `instruction` (including the desired entity types and relationship types), output all relationship triples `output_text` in the form of `(ent1, rel, ent2)` found within the `text`:

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
text="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"
output_text="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```

The meaning of knowledge graph completion is that, when given an input `miss_text` (a portion of the text is missing) and an `instruction`, the model is still able to complete the missing triples and output `output_text`. Here is an example:

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
miss_text="2006年，弗雷泽出战中国天津举行的女子水球世界杯。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"。
output_text="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```

Although the text "协助国家队夺得冠军" is not included in `miss_text`, the model can still complete the missing triples, i.e., it still needs to output `(弗雷泽,属于,国家队)(国家队,夺得,冠军)`.

## Data

The training dataset for the competition contains the following fields for each data entry:

|    Field    |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|     id      |                   Sample unique identifier                   |
|    text     | Model input text (need to extract all triples involved within) |
| instruction |   Instruction for the model to perform the extraction task   |
| output_text | Expected model output, in the form of output text composed of (ent1, relation, ent2) |
|     kg      |             Knowledge graph involved in the text             |
|   entity    | All entities involved in the kg, including entity type 'type' and entity name 'text' |
|  Relation   | All relationships involved in the kg, including relationship type 'type' and entities 'args' |

In the test set, only the three fields `id`, `instruction`, and `text` are included.

## Config Setup

This evaluation task is essentially a triple extraction (rte) task. Detailed parameters and configuration for using this module can be found in the [Environment and Data](#Requirement,-Data-and-Configuration) section above. The main parameter settings are as follows:

- Set `task` to `rte`, indicating a triple extraction task;
- Set `language` to `ch`, indicating that the task is based on Chinese data;
- Set `engine` to the desired OpenAI large model name (since the OpenAI GPT-4 API is not fully open, this module currently does not support the use of GPT-4 API);
- Set `text_input` to the `text` field in the dataset;
- Set `zero_shot` as needed; if set to `True`, examples for in-context learning need to be set in the `/data/rte_ch.json` file in a specific format;
- Set `instruction` to the `instruction` field in the dataset; if set to `None`, the default instruction for the module will be used;
- Set `labels` to the entity types, or leave it empty;

Other parameters can be left at their default values.

## Run and Example

After setting the parameters, simply run the `run.py` file:

```shell
>> python run.py
```

Input and output examples for making predictions using ChatGPT:

| Input                                                        | Output                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| task="rte"<br/>language="ch"<br/>engine="gpt-3.5-turbo"<br/>text_input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"<br/>instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答" | \[\[弗雷泽,获奖,铜牌\],\[女子水球世界杯,举办地点,天津\],\[弗雷泽,属于,国家队\],\[弗雷泽,国家,澳大利亚\],\[弗雷泽,参加,北京奥运会女子水球比赛\],\[中国,包含行政领土,天津\],\[中国,邦交国,澳大利亚\],\[北京奥运会女子水球比赛,举办地点,北京\],\[女子水球世界杯,体育运动,水球\],\[国家队,夺得,冠军)\] |

## Baseline Results

We conducted a simple 5-shot in-context learning evaluation on the CCKS dataset using **ChatGPT**, and the results are shown in the table below:

|              Metric               | Result |
| :-------------------------------: | :----: |
|                F1                 | 0.3995 |
|             Rougen_2              | 0.7730 |
| score</br>(0.5\*F1+0.5\*Rougen_2) | 0.5863 |

# CodeKGC-Code Language Models for Knowledge Graph Construction

To better address Relational Triple Extraction (rte) task in Knowledge Graph Construction, we have designed code-style prompts to model the structure of  Relational Triple, and used Code-LLMs to generate more accurate predictions. The key step of code-style prompt construction is to transform (text, output triples) pairs into semantically equivalent program language written in Python.

<div align=center>
<img src="./codekgc/codekgc_figure.png" width="85%" height="75%" />
</div>

## Data and Configuration

- Data

  The example data of `conll04` dataset is stored in the `codekgc/data` folder. The entire data is available in [here](https://drive.google.com/drive/folders/1vVKJIUzK4hIipfdEGmS0CCoFmUmZwOQV?usp=share_link). Users can customize your own data, but it is necessary to follow the given data format.

- Configuration

  The `codekgc/config.json` file contains the set parameters. The parameters required to load files and call the openai models are passed in through this file.

  Descriptions of these parameters are as follows:

  - `schema_path` defines the file path of the schema prompt. Schema prompt contains the pre-defined Python classes including **Relation** class, **Entity** class, **Triple** class and **Extract** class.

    The data format of schema prompt is as follows:

    ```python
    from typing import List
    class Rel:
        def __init__(self, name: str):
            self.name = name
    class Work_for(Rel):
    ...
    class Entity:
        def __init__(self, name: str):
            self.name = name
    class person(Entity):
    ...
    class Triple:
        def __init__(self, head: Entity, relation: Rel, tail: Entity):
            self.head = head
            self.relation = relation
            self.tail = tail
    class Extract:
        def __init__(self, triples: List[Triple] = []):
            self.triples = triples
    ```

  - `ICL_path` defines the file path of in-context examples.

    The data format of ICL prompt is as follows:

    ```python
    """ In 1856 , the 28th President of the United States , Thomas Woodrow Wilson , was born in Staunton , Va . """
    extract = Extract([Triple(person('Thomas Woodrow Wilson'), Rel('Live in'), location('Staunton , Va')),])
    ...
    ```

  - `example_path` defines the file path of the test example in conll04 dataset.

  - `openai_key` is your api key of openai.

  - `engine`, `temperature`, `max_tokens`, `n`... are the parameters required to pass in to call the openai api.

## Run and Examples

Once the parameters are set, you can directly run the `codekgc.py`：

```shell
>> cd codekgc
>> python codekgc.py
```

Below are input and output examples for Relational Triple Extraction (rte) task using code-style prompts:

**Input**:

```python
from typing import List
class Rel:
...(schema prompt)

""" In 1856 , the 28th President..."""
extract = Extract([Triple(person('Thomas Woodrow Wilson'), Rel('Live in'), location('Staunton , Va')),])
...(in-context examples)

""" Boston University 's Michael D. Papagiannis said he believes the crater was created 100 million years ago when a 50-mile-wide meteorite slammed into the Earth . """
```

**Output**:

```python
extract = Extract([Triple(person('Michael D. Papagiannis'), Rel('Work for'), organization('Boston University')),])
```
## Citation
If you use the code, please cite the following paper:

```bibtex
@article{DBLP:journals/corr/abs-2304-09048,
  author       = {Zhen Bi and
                  Jing Chen and
                  Yinuo Jiang and
                  Feiyu Xiong and
                  Wei Guo and
                  Huajun Chen and
                  Ningyu Zhang},
  title        = {CodeKGC: Code Language Model for Generative Knowledge Graph Construction},
  journal      = {CoRR},
  volume       = {abs/2304.09048},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2304.09048},
  doi          = {10.48550/arXiv.2304.09048},
  eprinttype    = {arXiv},
  eprint       = {2304.09048},
  timestamp    = {Mon, 24 Apr 2023 15:03:18 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2304-09048.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
