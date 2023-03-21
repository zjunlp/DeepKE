<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README.md">English</a> | 简体中文</a> </b>
</p>

## 目录

- [目录](#目录)
- [使用大语言模型进行信息抽取](#使用大语言模型进行信息抽取)
  - [环境与数据](#环境与数据)
  - [使用与示例](#使用与示例)
- [使用大语言模型进行数据增强](#使用大语言模型进行数据增强)
  - [参数设置](参数设置)

# 使用大语言模型进行信息抽取

## 环境与数据

- 环境配置

  Deepke大模型模块使用[EasyInstruct](https://github.com/zjunlp/EasyInstruct)工具

  ```shell
  >> pip install easyinstruct
  >> pip install hydra-core
  ```

- 数据

  这里的数据指的是用于in-context learning的examples数据，放在`data`文件夹中，其中的`.json`文件是各种任务默认的examples数据，用户可以自定义其中的example，但需要遵守给定的数据格式。

- 参数配置

  `conf`文件夹保存所设置的参数。调用大模型接口所需要的参数都通过此文件夹中文件传入。

  - 在命名实体识别任务(ner)中，`text_input`参数为预测文本；`domain`为预测文本所属领域，可为空；`labels`为实体标签集，如无自定义的标签集，该参数可为空。

  - 在关系抽取任务(re)中，`text_input`参数为文本；`domain`为文本所属领域，可为空；`labels`为关系类型标签集，如无自定义的标签集，该参数可为空；`head_entity`和`tail_entity`为待预测关系的头实体和尾实体；`head_type`和`tail_type`为待预测关系的头尾实体类型。

  - 在事件抽取任务(ee)中，`text_input`参数为待预测文本；`domain`为预测文本所属领域，可为空。

  - 在三元组抽取任务(rte)中，`text_input`参数为待预测文本；`domain`为预测文本所属领域，可为空。

  - 其他参数的具体含义：
    - `task`参数用于指定任务类型，其中`ner`表示命名实体识别任务，`re`表示关系抽取任务`ee`表示事件抽取任务，`rte`表示三元组抽取任务；
    - `language`表示任务的语言，`en`表示英文抽取任务，`ch`表示中文抽取任务；
    - `engine`表示所用的大模型名称，要与OpenAI API规定的模型名称一致；
    - `api_key`是用户的API密钥；
    - `zero_shot`表示是否为零样本设定，为`True`时表示只使用instruction提示模型进行信息抽取，为`False`时表示使用in-context的形式进行信息抽取；
    - `instruction`参数用于规定用户自定义的提示指令，当为空时采用默认的指令；
    - `data_path`表示in-context examples的存储目录，默认为`data`文件夹。

## 使用与示例

设定好参数后，直接运行`run.py`文件即可:

```shell
>> python run.py
```

以下是不同任务的输入输出示例：

|     任务     |                           输入文本                           |                             输出                             |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 命名实体识别 | 《红楼梦》是中央电视台和中国电视剧制作中心根据中国古典文学名著《红楼梦》摄制于1987年的一部古装连续剧，由王扶林导演，周汝昌、王蒙、周岭等多位红学家参与制作。 | [{'E': 'TV Series', 'W': '红楼梦'}, {'E': 'Director', 'W': '王扶林'}, {'E': 'Actor', 'W': '周汝昌'}, {'E': 'Actor', 'W': '王蒙'}, {'E': 'Actor', 'W': '周岭'}] |
|   关系抽取   | 孔正锡，导演，2005年以一部温馨的爱情电影《长腿叔叔》敲开电影界大门<br>(需指定头尾实体和实体类型) |                             导演                             |
|   事件抽取   | 历经4小时51分钟的体力、意志力鏖战，北京时间9月9日上午纳达尔在亚瑟·阿什球场，以7比5、6比3、5比7、4比6和6比4击败赛会5号种子俄罗斯球员梅德韦杰夫，夺得了2019年美国网球公开赛男单冠军。 | event_list: [event_type: [arguments: [role: 纳达尔, argument: 夺得2019年美国网球公开赛男单冠军], [role: 梅德韦杰夫, argument: 被纳达尔击败]], [event_type: [arguments: [role: 纳达尔, argument: 以7比5、6比3、5比7、4比6和6比4击败梅德韦杰夫]], [event_type: [arguments: [role: 纳达尔, argument: 历经4小时51分钟的体力、意志力鏖战]], [event_type: [arguments: [role: 纳达尔, argument: 在亚瑟·阿什球场]], [event_type: [arguments: [role: 梅德韦杰夫, argument: 赛会5号种子俄罗斯球员]]]] |
| 联合关系抽取 |  《没有你的夜晚》是歌手欧阳菲菲演唱的歌曲，出自专辑《拥抱》  | [['《没有你的夜晚》', '演唱者', '欧阳菲菲'], ['《没有你的夜晚》', '出自专辑', '《拥抱》']] |

# 使用大语言模型进行数据增强

为了弥补少样本场景下关系抽取有标签数据的缺失, 我们设计带有数据样式描述的提示，用于指导大型语言模型根据已有的少样本数据自动地生成更多的有标签数据。

## 参数设置

- `task`设置为`da`；
- `text_input`设置为要增强的关系标签，比如`org:founded_by`；
- `zero_shot`设为`False`，并在`data`文件夹下`da`对应的文件中设置少样本样例；
- `labels`中可以指定头尾实体的标签范围。

如下为一个数据增强的`prompt`示例:

```PYTHON
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

