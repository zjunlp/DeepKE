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
- [CCKS2023](#CCKS2023)

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

# CCKS2023

see [here](https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md)
