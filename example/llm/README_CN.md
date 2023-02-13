

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README.md">English</a> | 简体中文</a> </b>
</p>

## 目录
- [目录](#目录)

- [使用大型语言模型进行命名实体识别、事件抽取以及实体关系联合抽取](#使用大型语言模型进行命名实体识别、事件抽取以及实体关系联合抽取)
  - [环境和数据集](#环境和数据集)
  - [使用示例](#使用示例)

- [使用大型语言模型进行关系抽取](#使用大型语言模型进行关系抽取)
  - [环境和数据集](#环境和数据集)
  - [提示](#提示)
  - [上下文学习](#上下文学习)
  - [使用大型语言模型生成数据](#使用大型语言模型生成数据)

本项目中部分代码源自[Promptify](https://github.com/promptslab/Promptify)，在此十分感谢Promptify团队。

# 使用大型语言模型进行命名实体识别、事件抽取以及实体关系联合抽取

## 环境和数据集
- 环境配置
  
  使用GPT-3需要OpenAI API (key) 
    ```shell
    >> pip install openai
    >> pip install jinja2
    >> pip install hydra-core
    ```

- 数据集及配置参数
  data文件夹中，所给json文件为数据所要求的格式。
  
  `conf`文件夹保存所设置的参数。调用GPT3接口所需要的参数都通过此文件夹中文件传入。
  
- 在命名实体识别任务中，`text_input`参数为预测文本，`examples`为少样本或零样本示例，可为空，`domain`为预测文本所属领域，可为空，`label`为实体标签集，也可为空。
  
- 在事件抽取任务中，`text_input`参数为预测文本，`examples`为少样本或零样本示例，可为空，`domain`为预测文本所属领域，也可为空。
 
- 在实体关系联合抽取任务中，`text_input`参数为预测文本，`examples`为少样本或零样本示例，可为空，`domain`为预测文本所属领域，也可为空。


## 使用示例
  |                           任务                           |           输入文本           |    输出    |     
  | :----------------------------------------------------------: | :------------------------: | :------------: |
  | 命名实体识别 |            《红楼梦》是中央电视台和中国电视剧制作中心根据中国古典文学名著《红楼梦》摄制于1987年的一部古装连续剧，由王扶林导演，周汝昌、王蒙、周岭等多位红学家参与制作。            |      [{'E': 'TV Series', 'W': '红楼梦'}, {'E': 'Director', 'W': '王扶林'}, {'E': 'Actor', 'W': '周汝昌'}, {'E': 'Actor', 'W': '王蒙'}, {'E': 'Actor', 'W': '周岭'}]      |          
  | 事件抽取 | 历经4小时51分钟的体力、意志力鏖战，北京时间9月9日上午纳达尔在亚瑟·阿什球场，以7比5、6比3、5比7、4比6和6比4击败赛会5号种子俄罗斯球员梅德韦杰夫，夺得了2019年美国网球公开赛男单冠军。 |       event_list: [event_type: [arguments: [role: 纳达尔, argument: 夺得2019年美国网球公开赛男单冠军], [role: 梅德韦杰夫, argument: 被纳达尔击败]], [event_type: [arguments: [role: 纳达尔, argument: 以7比5、6比3、5比7、4比6和6比4击败梅德韦杰夫]], [event_type: [arguments: [role: 纳达尔, argument: 历经4小时51分钟的体力、意志力鏖战]], [event_type: [arguments: [role: 纳达尔, argument: 在亚瑟·阿什球场]], [event_type: [arguments: [role: 梅德韦杰夫, argument: 赛会5号种子俄罗斯球员]]]]     |  
  | 联合关系抽取 |           《没有你的夜晚》是歌手欧阳菲菲演唱的歌曲，出自专辑《拥抱》           | [['《没有你的夜晚》', '演唱者', '欧阳菲菲'], ['《没有你的夜晚》', '出自专辑', '《拥抱》']]|     


# 使用大型语言模型进行关系抽取


## 环境和数据集
- 环境配置
  
  使用GPT-3需要OpenAI API (key) 
    ```shell
    >> pip install openai
    ```
- 数据集
  - [TACRED](https://nlp.stanford.edu/projects/tacred/)
  - [TACREV](https://github.com/DFKI-NLP/tacrev)
  - [RE-TACRED](https://github.com/gstoica27/Re-TACRED)


## 提示
![prompt](LLM.png)

## 上下文学习
我们通过提供带有示例的提示给大型语言模型，使用上下文学习引导大型语言模型理解关系抽取任务。如上图所示，我们设计了两种提示：(1) **TEXT PROMPT** 仅包含关系抽取任务最基本的必要信息；(2) **INSTRUCT PROMPT** 包含关系抽取任务相关的指令。同时，为了实现更好的抽取性效果，实体类型可以作为schemas信息加入到提示中。

请使用以下脚本实现K-shot示例的上下文学习：

```shell
>> python gpt3ICL.py -h
    usage: gpt3ICL.py [-h] --api_key API_KEY --train_path TRAIN_PATH --test_path TEST_PATH --output_success OUTPUT_SUCCESS --output_nores OUTPUT_NORES --prompt {text,text_schema,instruct,instruct_schema} [--k K]

    optional arguments:
      -h, --help            show this help message and exit
      --api_key API_KEY, -ak API_KEY
      --train_path TRAIN_PATH, -tp TRAIN_PATH
                            The path of training / demonstration data.
      --test_path TEST_PATH, -ttp TEST_PATH
                            The path of test data.
      --output_success OUTPUT_SUCCESS, -os OUTPUT_SUCCESS
                            The output directory of successful ICL samples.
      --output_nores OUTPUT_NORES, -on OUTPUT_NORES
                            The output directory of failed ICL samples.
      --prompt {text,text_schema,instruct,instruct_schema}
      --k K                 k-shot demonstrations
```

## 使用大型语言模型生成数据
为了弥补少样本场景下关系抽取有标签数据的缺失, 我们设计带有数据样式描述的提示，用于指导大型语言模型自动地生成更多的有标签数据根据已有的少样本数据，如上图最后一条提示所示。

请使用一下脚本生成数据:
```shell
>> python gpt3DA.py -h
  usage: gpt3DA.py [-h] --api_key API_KEY --demo_path DEMO_PATH --output_dir OUTPUT_DIR --dataset {tacred,tacrev,retacred} [--k K]

  optional arguments:
    -h, --help            show this help message and exit
    --api_key API_KEY, -ak API_KEY
    --demo_path DEMO_PATH, -dp DEMO_PATH
                          The directory of demonstration data.
    --output_dir OUTPUT_DIR
                          The output directory of generated data.
    --dataset {tacred,tacrev,retacred}
    --k K                 k-shot demonstrations
```
您可以使用大模型生成的数据和原始数据，基于常规关系抽取的代码[standard-re](https://github.com/zjunlp/DeepKE/tree/main/example/re/standard)进行模型训练。
