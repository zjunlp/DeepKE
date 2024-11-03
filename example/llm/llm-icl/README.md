# deepke llm readme 结构



## 1 整体介绍

1里面加致谢

1 里 加一个动图 就是到时候做的前端演示系统



### 1-?: 参数设置


```yaml
cwd:
engine:
model_id:
api_key:
base_url:
temperature:
top_p:
max_tokens:
stop:
task:
language:
in_context:
instruction:
data_path:
text_input:
domain:
labels:
head_entity:
head_type:
tail_entity:
tail_type:
```

全参数表：

| 分类 | 参数名称    | 类型      | 意义                                                         | 限定                                                         |
| ---- | ----------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 系统 | cwd         | str       | 当前工作目录。                                               |                                                              |
| 设定 | engine      | str       | 表示所用的大模型名称。                                       | ["LLaMA", "Qwen", "MiniCPM", "ChatGLM",  "ChatGPT", "DeepSeek"] |
| 设定 | api_key     | str       | 用户的API密钥。                                              |                                                              |
| 设定 | base_url    | str       |                                                              |                                                              |
| 设定 | temperature | float     |                                                              |                                                              |
| 设定 | top_p       | float     |                                                              |                                                              |
| 设定 | max_tokens  | int       |                                                              |                                                              |
| 设定 | stop        | ?         | ?                                                            |                                                              |
| 设定 | task        | str       | 参数用于指定任务类型，其中`ner`表示命名实体识别任务，`re`表示关系抽取任务`ee`表示事件抽取任务，`rte`表示三元组抽取任务。 | ["ner", "re", "ee", "rte", "da"]                             |
| 设定 | language    | str       | 表示任务的语言。                                             | ["en", "ch"]                                                 |
| 设定 | in_context  | bool      | 是否为零样本设定，为`False`时表示只使用instruction提示模型进行信息抽取，为`True`时表示使用in-context的形式进行信息抽取； |                                                              |
| 设定 | instruction | str       | 规定用户自定义的提示指令，当为空时采用默认的指令；           | （不建议使用）in_context == False时可以自设examples          |
| 设定 | data_path   | str       | 表示in-context examples的存储目录，默认为`data`文件夹。      | in_context == True                                           |
| 任务 | text_input  | str       | 在命名实体识别任务(`ner`)中，`text_input`参数为预测文本；在关系抽取任务(`re`)中，`text_input`参数为文本；在事件抽取任务(`ee`)中，`text_input`参数为待预测文本；在三元组抽取任务(`rte`)中，`text_input`参数为待预测文本。 | ner、re、ee、rte、da                                         |
| 任务 | domain      | str       | 在命名实体识别任务(`ner`)中，`domain`为预测文本所属领域，可为空；在关系抽取任务(`re`)中，`domain`为文本所属领域，可为空；在事件抽取任务(`ee`)中，`domain`为预测文本所属领域，可为空；在三元组抽取任务(`rte`)中，`domain`为预测文本所属领域，可为空。 | ner、re、ee、rte                                             |
| 任务 | labels      | List[str] | 在命名实体识别任务(`ner`)中，`labels`为实体标签集，如无自定义的标签集，该参数可为空；在关系抽取任务(`re`)中，`labels`为关系类型标签集，如无自定义的标签集，该参数可为空。`da`中`lables`为头尾实体被预先分类的类型。 | ner、re、da                                                  |
| 任务 | head_entity | str       | 待预测关系的头实体和尾实体；                                 | re                                                           |
| 任务 | tail_entity | str       | 待预测关系的头实体和尾实体；                                 | re                                                           |
| 任务 | head_type   | str       | 待预测关系的头尾实体类型。                                   | re                                                           |
| 任务 | tail_type   | str       | 待预测关系的头尾实体类型。                                   | re                                                           |



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





## 2 快速启动 包含prompt 例子一个和sft 例子一个 用qwen

在Python中，如果在`config.yaml`中未指定某个参数，而该参数在代码中被引用，则通常会导致KeyError，除非在代码中为其提供了默认值或处理了不存在的情况。因此，最好在配置文件中写全所有参数，即便一些参数未明确需要使用。这样可以保证运行的稳定性，减少出错的可能性。

以下是每个`config.yaml`的全参数示例，确保所有配置文件中都包含每个可能使用的参数，并为未使用的参数指定默认值（通常为`null`）。



```yaml
cwd: null
engine: "ChatGPT"
model_id: "gpt-3.5-turbo"
api_key: "sk-7kVvoLKWePIdvvirGUXY80MVqAsofr3f37kqqHUw7Pnmil3u"
base_url: "https://api.chatanywhere.tech/v1"
temperature: 0.3
top_p: 0.9
max_tokens: 100
stop: null
task: "ner"
language: "ch"
in_context: true
#instruction:
data_path: "data"
text_input: "比尔·盖茨是美国企业家、软件工程师、慈善家、微软公司创始人、中国工程院外籍院士。曾任微软董事长、CEO和首席软件设计师。"
domain: "人物"
labels: ["头衔", "任职"]
head_entity:
head_type:
tail_entity:
tail_type:

```



```yaml
cwd: null
engine: "ChatGPT"
model_id: "gpt-3.5-turbo"
api_key: "sk-7kVvoLKWePIdvvirGUXY80MVqAsofr3f37kqqHUw7Pnmil3u"
base_url: "https://api.chatanywhere.tech/v1"
temperature: 0.3
top_p: 0.9
max_tokens: 100
stop: null
task: "re"
language: "ch"
in_context: true
#instruction:
data_path: "data"
text_input: "艾伦就读于湖滨中学，在这里他遇到和自己一样对计算机编程痴迷的盖茨。"
domain: "情感识别"
labels: ["感情"]
head_entity: "艾伦"
head_type: "人物"
tail_entity: "计算机编程"
tail_type: "事件"

```



```yaml
cwd: null
engine: "ChatGPT"
model_id: "gpt-3.5-turbo"
api_key: "sk-7kVvoLKWePIdvvirGUXY80MVqAsofr3f37kqqHUw7Pnmil3u"
base_url: "https://api.chatanywhere.tech/v1"
temperature: 0.3
top_p: 0.9
max_tokens: 300
stop: null
task: "ee"
language: "ch"
in_context: true
#instruction:
data_path: "data"
text_input: "2007年11月6日，阿里巴巴正式以港币13.5元在香港联合交易所挂牌上市，股票代码为“1688 HK”。阿里巴巴上市开盘价30港元，较发行价提高122%。融资116亿港元，创下中国互联网公司融资规模之最。"
domain: "财经"
labels:
head_entity:
head_type:
tail_entity:
tail_type:

```



```yaml
cwd: null
engine: "ChatGPT"
model_id: "gpt-3.5-turbo"
api_key: "sk-7kVvoLKWePIdvvirGUXY80MVqAsofr3f37kqqHUw7Pnmil3u"
base_url: "https://api.chatanywhere.tech/v1"
temperature: 0.3
top_p: 0.9
max_tokens: 150
stop: null
task: "rte"
language: "ch"
in_context: true
#instruction: "我让你完成RTE任务。"
data_path: "data"
text_input: "卢浮宫始建于1204年，位于法国巴黎市中心的塞纳河北岸。"
domain: "地理"
#labels: [""]
#head_entity:
#head_type:
#tail_entity:
#tail_type:

```



```yaml
cwd: null
engine: "ChatGPT"
model_id: "gpt-3.5-turbo"
api_key: "sk-7kVvoLKWePIdvvirGUXY80MVqAsofr3f37kqqHUw7Pnmil3u"
base_url: "https://api.chatanywhere.tech/v1"
temperature: 0.3
top_p: 0.9
max_tokens: 300
stop: null
task: "da"
language: "ch"
in_context: true
#instruction: ""
data_path: "data"
text_input: "创立"
domain:
labels: ["人物", "公司"]
#head_entity:
#head_type:
#tail_entity:
#tail_type:

```



## 3  具体prompt 例子 包含 qwen llama chatgpt 和自定义训好的模型llama吧   里面需要有脚本 和输入输出例子

3里单独加一个 以qwen为例 输入不同类型的数据 





## 4 一些注意事项 如常见错误





## 5 limitation



