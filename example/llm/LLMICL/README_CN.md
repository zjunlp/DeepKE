# 环境与数据
<a id="requirements"></a>
- 环境配置

  ```shell
  >> pip install git+https://github.com/zjunlp/EasyInstruct
  >> pip install hydra-core
  ```

- 数据

  这里的数据指的是用于`In-Context Learning`的examples数据，放在`data`文件夹中，其中的`.json`文件是各种任务默认的examples数据，用户可以自定义其中的example，但需要遵守给定的数据格式。

- 参数配置

  `conf`文件夹保存所设置的参数。调用大模型接口所需要的参数都通过此文件夹中文件传入。

  - 在命名实体识别任务(`ner`)中，`text_input`参数为预测文本；`domain`为预测文本所属领域，可为空；`labels`为实体标签集，如无自定义的标签集，该参数可为空。

  - 在关系抽取任务(`re`)中，`text_input`参数为文本；`domain`为文本所属领域，可为空；`labels`为关系类型标签集，如无自定义的标签集，该参数可为空；`head_entity`和`tail_entity`为待预测关系的头实体和尾实体；`head_type`和`tail_type`为待预测关系的头尾实体类型。

  - 在事件抽取任务(`ee`)中，`text_input`参数为待预测文本；`domain`为预测文本所属领域，可为空。

  - 在三元组抽取任务(`rte`)中，`text_input`参数为待预测文本；`domain`为预测文本所属领域，可为空。

  - 其他参数的具体含义：
    - `task`参数用于指定任务类型，其中`ner`表示命名实体识别任务，`re`表示关系抽取任务`ee`表示事件抽取任务，`rte`表示三元组抽取任务；
    - `language`表示任务的语言，`en`表示英文抽取任务，`ch`表示中文抽取任务；
    - `engine`表示所用的大模型名称，要与OpenAI API规定的模型名称一致；
    - `api_key`是用户的API密钥；
    - `zero_shot`表示是否为零样本设定，为`True`时表示只使用instruction提示模型进行信息抽取，为`False`时表示使用in-context的形式进行信息抽取；
    - `instruction`参数用于规定用户自定义的提示指令，当为空时采用默认的指令；
    - `data_path`表示in-context examples的存储目录，默认为`data`文件夹。


# 使用大语言模型进行信息抽取

### 使用与示例


我们使用[EasyInstruct](https://github.com/zjunlp/EasyInstruct)工具(一个简单易用的指导大语言模型的框架) 完成这一任务, 环境与数据请参考[第一章](#requirements)

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

### 参数设置

- `task`设置为`da`；
- `text_input`设置为要增强的关系标签，比如`org:founded_by`；
- `zero_shot`设为`False`，并在`data`文件夹下`da`对应的文件中设置少样本样例；
- `labels`中可以指定头尾实体的标签范围。


### 使用与示例

我们使用[EasyInstruct](https://github.com/zjunlp/EasyInstruct)工具(一个简单易用的指导大语言模型的框架) 完成这一任务, 环境与数据请参考[第一章](#requirements)

设定好参数后，直接运行`run.py`文件即可:

```shell
>> python run.py
```

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



# 使用大语言模型完成CCKS2023指令驱动的知识图谱构建

下面是[CCKS2023指令驱动的自适应知识图谱构建评测任务](https://tianchi.aliyun.com/competition/entrance/532080/introduction?spm=5176.12281957.0.0.4c885d9b2YX9Nu)关于*ChatGPT/GPT-4*的baseline说明。

### 任务目标

根据用户输入的指令抽取相应类型的实体和关系，构建知识图谱。其中可能包含知识图谱补全任务，即任务需要模型在抽取实体关系三元组的同时对缺失三元组进行补全。

以下是一个**知识图谱构建任务**例子，输入一段文本`input`和`instruction`（包括想要抽取的实体类型和关系类型），以`(ent1,rel,ent2)`的形式输出`input`中包含的所有关系三元组`output`：

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```



### 数据


下面是一些现成的处理后的数据：

| 名称                  | 下载                                                                                                                     | 数量     | 描述                                                                                                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KnowLM-IE.json       | [Google drive](https://drive.google.com/file/d/1hY_R6aFgW4Ga7zo41VpOVOShbTgBqBbL/view?usp=sharing) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)      | 281860 | [InstructIE](https://arxiv.org/abs/2305.11527) 中提到的数据集                                                                                     |
| train.json, valid.json          | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing)                     | 5000   | [CCKS2023 开放环境下的知识图谱构建与补全评测任务一：指令驱动的自适应知识图谱构建](https://tianchi.aliyun.com/competition/entrance/532080/introduction) 中的初赛训练集及测试集，从 KnowLM-IE.json 中随机选出的 |


`KnowLM-IE.json`：包含 `'id'`(唯一标识符)、`'cate'`(文本主题)、`'instruction'`(抽取指令)、`'input'`(输入文本)、`'output'`(输出文本)字段、`'relation'`(三元组)字段，可以通过`'relation'`自由构建抽取的指令和输出，`'instruction'`有16种格式(4种prompt * 4种输出格式)，`'output'`是按照`'instruction'`中指定的输出格式生成的文本。。


`train.json`：字段含义同`train.json`，`'instruction'`、`'output'`都只有1种格式，也可以通过`'relation'`自由构建抽取的指令和输出。

`valid.json`：字段含义同`train.json`，但是经过众包标注，更加准确。


以下是各字段的说明：

|    字段     |                          说明                          |
| :---------: | :----------------------------------------------------: |
|     id      |                     唯一标识符                     |
|    cate     |     文本input对应的主题(共12种)                          |
|    input    |    模型输入文本（需要抽取其中涉及的所有关系三元组）    |
| instruction |                 模型进行抽取任务的指令                 |
| output      | 模型期望输出 |
| relation    |                  input中涉及的关系三元组(head, relation, tail)               |


更多有关数据处理和数据格式的信息请参考[InstructKGC/kg2instruction](../InstructKGC/kg2instruction/README_CN.md)




### 参数设置

该评测任务本质上是一个三元组抽取(rte)任务，使用该模块时详细参数与配置可见上文中的[环境与数据](#环境与数据)部分。主要的参数设置如下：

- `task`设置为`rte`，表示三元组抽取任务；
- `language`设置为`ch`，表示该任务是中文数据；
- `engine`设置为想要使用的OpenAI大模型名称(由于OpenAI GPT-4 API未完全开放，本模块目前暂不支持GPT-4 API的使用)；
- `text_input`设置为数据集中的`input`文本；
- `zero_shot`可根据需要设置，如设置为`False`，需要在`/data/rte_ch.json`文件中按照特定格式设置in-context learning所需的examples；
- `instruction`可设置为数据集中的`instruction`字段，如果为`None`则表示使用模块默认的指令；
- `labels`可设置为实体类型，也可为空；

其它参数默认即可。

我们为CCKS2023比赛数据格式提供了一个转化脚本, `LLMICL/ccks2023_convert.py`



### 使用与示例

我们使用[EasyInstruct](https://github.com/zjunlp/EasyInstruct)工具(一个简单易用的指导大语言模型的框架) 完成这一任务, 环境与数据请参考[第一章](#requirements)

设定好参数后，直接运行`run.py`文件即可:

```shell
>> python run.py
```

使用ChatGPT进行预测的输入输出示例：

| 输入                                                         | 输出                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| task="rte"<br/>language="ch"<br/>engine="gpt-3.5-turbo"<br/>text_input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"<br/>instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答" | \[\[弗雷泽,获奖,铜牌\],\[女子水球世界杯,举办地点,天津\],\[弗雷泽,属于,国家队\],\[弗雷泽,国家,澳大利亚\],\[弗雷泽,参加,北京奥运会女子水球比赛\],\[中国,包含行政领土,天津\],\[中国,邦交国,澳大利亚\],\[北京奥运会女子水球比赛,举办地点,北京\],\[女子水球世界杯,体育运动,水球\],\[国家队,夺得,冠军)\] |

### 基线结果

我们基于**ChatGPT**在CCKS数据集上进行了5-shot的in-context learning简单评测，结果如下表所示：

|               指标                |  结果  |
| :-------------------------------: | :----: |
|                F1                 | 0.3995 |
|             Rougen_2              | 0.7730 |
| score</br>(0.5\*F1+0.5\*Rougen_2) | 0.5863 |

