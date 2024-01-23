# Requirement, Data and Configuration
<a id="requirements"></a>
- Requirement

  The LLM module of Deepke calls the [EasyInstruct](https://github.com/zjunlp/EasyInstruct) tookit(An Easy-to-use Framework to Instruct Large Language Models).

  ```
  >> pip install git+https://github.com/zjunlp/EasyInstruct
  >> pip install hydra-core
  ```

- Data

  The data here refers to the examples data used for in-context learning, which is stored in the `data` folder. The `.json` files in it are the default examples data for various tasks. Users can customize the examples in them, but they need to follow the given data format.

- Configuration

  The `conf` folder stores the set parameters. The parameters required to call the GPT3 interface are passed in through the files in this folder.

  - In the Named Entity Recognition (`ner`) task, `text_input` parameter is the prediction text; `domain` is the domain of the prediction text, which can be empty; `label` is the entity label set, which can also be empty. 

  - In the Relation Extraction (`re`) task, `text_input` parameter is the text; `domain` indicates the domain to which the text belongs, and it can be empty; `labels` is the set of relationship type labels. If there is no custom label set, this parameter can be empty; `head_entity` and `tail_entity` are the head entity and tail entity of the relationship to be predicted, respectively; `head_type` and `tail_type` are the types of the head and tail entities to be predicted in the relationship.

  - In the Event Extraction (`ee`) task, `text_input` parameter is the prediction text; `domain` is the domain of the prediction text, which can also be empty. 

  - In the Relational Triple Extraction (`rte`) task, `text_input` parameter is the prediction text; `domain` is the domain of the prediction text, which can also be empty.

  - The specific meanings of other parameters are as follows:
    - `task` parameter is used to specify the task type, where `ner` represents named entity recognition task, `re` represents relation extraction task, `ee` represents event extraction task, and `rte` represents triple extraction task;
    - `language` indicates the language of the task, where `en` represents English extraction tasks, and `ch` represents Chinese extraction tasks;
    - `engine` indicates the name of the large model used, which should be consistent with the model name specified by the OpenAI API;
    - `api_key` is the user's API key;
    - `zero_shot` indicates whether zero-shot setting is used. When it is set to `True`, only the instruction prompt model is used for information extraction, and when it is set to `False`, in-context form is used for information extraction;
    - `instruction` parameter is used to specify the user-defined prompt instruction, and the default instruction is used when it is empty;
    - `data_path` indicates the directory where in-context examples are stored, and the default is the `data` folder.



# IE with Large Language Models

We use the [EasyInstruct](https://github.com/zjunlp/EasyInstruct) tool, a user-friendly framework for instructing large language models, to complete this task. Please refer to [Chapter 1](#requirements) for the environment and data. 

### Run and Examples

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


### Configuration

- Set `task` to `da`;
- Set `text_input` to the relationship label to be enhanced, such as `org:founded_by`;
- Set `zero_shot` to `False` and set the low-sample example in the corresponding file under the `data` folder for the `da` task;
- The range of entity labels can be specified in `labels`.

### Run and Examples

We use the [EasyInstruct](https://github.com/zjunlp/EasyInstruct) tool, a user-friendly framework for instructing large language models, to complete this task. Please refer to [Chapter 1](#requirements) for the environment and data. 

Once the parameters are set, you can directly run the `run.py`：

```
>> python run.py
```


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



# CCKS2023 Instruction-based Knowledge Graph Construction with Large Language Models


The following is a baseline description of the *ChatGPT/GPT-4* for the **Instruction-based Knowledge Graph Construction** task in the **[CCKS2023 Open Environment Knowledge Graph Construction and Completion Evaluation competition](https://tianchi.aliyun.com/competition/entrance/532080/introduction?spm=5176.12281957.0.0.4c885d9b2YX9Nu)**.

### Task Object

Extract relevant entities and relations according to user input instructions to construct a knowledge graph. This task may include knowledge graph completion, where the model is required to complete missing triples while extracting entity-relation triples.

Below is an example of a **Knowledge Graph Construction Task**. Given an input text `input` and an `instruction` (including the desired entity types and relationship types), output all relationship triples `output` in the form of `(ent1, rel, ent2)` found within the `input`:

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```



### Datasets

Here are some readily processed datasets:

| Name                   | Download                                                     | Quantity | Description                                                  |
| ---------------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| KnowLM-IE.json          | [Google drive](https://drive.google.com/file/d/1hY_R6aFgW4Ga7zo41VpOVOShbTgBqBbL/view?usp=sharing) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)  | 281,860  | Dataset mentioned in [InstructIE](https://arxiv.org/abs/2305.11527) |
| train.json, valid.json | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing) | 5,000    | Preliminary training set and test set for the task "Instruction-Driven Adaptive Knowledge Graph Construction" in [CCKS2023 Open Knowledge Graph Challenge](https://tianchi.aliyun.com/competition/entrance/532080/introduction), randomly selected from instruct_train.json |

`KnowLM-IE.json`: Contains `'id'` (unique identifier), `'cate'` (text category), `'instruction'` (extraction instruction), `'input'` (input text), `'output'` (output text) and `'relation'` (triples) fields, allowing for the flexible construction of extraction instructions and outputs through `'relation'`, `'instruction'` has 16 formats (4 prompts * 4 output formats), and `'output'` is generated according to the specified output format in `'instruction'`.

`train.json`: Same fields as `KnowLM-IE.json`, `'instruction'` and `'output'` have only one format, and extraction instructions and outputs can also be freely constructed through `'relation'`.

`valid.json`: Same fields as `train.json`, but with more accurate annotations achieved through crowdsourcing.


Here is an explanation of each field:

|    Field    |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|     id      |                   Unique identifier                   |
|    cate     |     text topic of input (12 topics in total)                        |
|    input    | Model input text (need to extract all triples involved within) |
| instruction |   Instruction for the model to perform the extraction task   |
|    output   | Expected model output |
|   relation  |             Relation triples(head, relation, tail) involved in the input             |


For more information on data processing and data formats, please refer to [../InstructKGC/kg2instruction](../InstructKGC/kg2instruction/README.md)



### Config Setup

This evaluation task is essentially a triple extraction (rte) task. Detailed parameters and configuration for using this module can be found in the [Environment and Data](#requirements) section above. The main parameter settings are as follows:

- Set `task` to `rte`, indicating a triple extraction task;
- Set `language` to `ch`, indicating that the task is based on Chinese data;
- Set `engine` to the desired OpenAI large model name (since the OpenAI GPT-4 API is not fully open, this module currently does not support the use of GPT-4 API);
- Set `text_input` to the `text` field in the dataset;
- Set `zero_shot` as needed; if set to `False`, examples for in-context learning need to be set in the `/data/rte_ch.json` file in a specific format;
- Set `instruction` to the `instruction` field in the dataset; if set to `None`, the default instruction for the module will be used;
- Set `labels` to the entity types, or leave it empty;

Other parameters can be left at their default values.

We have provided a conversion script for the CCKS2023 competition data format, `LLMICL/ccks2023_convert.py`

### Run and Example

We use the [EasyInstruct](https://github.com/zjunlp/EasyInstruct) tool, a user-friendly framework for instructing large language models, to complete this task. Please refer to [Chapter 1](#requirements) for the environment and data. 

After setting the parameters, simply run the `run.py` file:

```shell
>> python run.py
```

Input and output examples for making predictions using ChatGPT:

| Input                                                        | Output                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| task="rte"<br/>language="ch"<br/>engine="gpt-3.5-turbo"<br/>text_input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"<br/>instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答" | \[\[弗雷泽,获奖,铜牌\],\[女子水球世界杯,举办地点,天津\],\[弗雷泽,属于,国家队\],\[弗雷泽,国家,澳大利亚\],\[弗雷泽,参加,北京奥运会女子水球比赛\],\[中国,包含行政领土,天津\],\[中国,邦交国,澳大利亚\],\[北京奥运会女子水球比赛,举办地点,北京\],\[女子水球世界杯,体育运动,水球\],\[国家队,夺得,冠军)\] |

### Baseline Results

We conducted a simple 5-shot in-context learning evaluation on the CCKS dataset using **ChatGPT**, and the results are shown in the table below:

|              Metric               | Result |
| :-------------------------------: | :----: |
|                F1                 | 0.3995 |
|             Rougen_2              | 0.7730 |
| score</br>(0.5\*F1+0.5\*Rougen_2) | 0.5863 |

