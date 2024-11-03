# deepke llm readme 结构

## 1 整体介绍

1里面加致谢

1 里 加一个动图 就是到时候做的前端演示系统

### 1-?: 参数设置

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

## 3  具体prompt 例子 包含 qwen llama chatgpt 和自定义训好的模型llama吧   里面需要有脚本 和输入输出例子

3里单独加一个 以qwen为例 输入不同类型的数据 

## 4 一些注意事项 如常见错误 5 limitation


