
<p align="center">
    <a> <img src="assets/LLM_logo.png" width="400"/></a>
<p>
<p align="center">
    <a href="http://deepke.zjukg.cn">
        <img alt="Documentation" src="https://img.shields.io/badge/demo-website-blue">
    </a>
    <a href="https://pypi.org/project/deepke/#files">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/deepke">
    </a>
    <a href="https://github.com/zjunlp/DeepKE/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/zjunlp/deepke">
    </a>
    <a href="http://zjunlp.github.io/DeepKE">
        <img alt="Documentation" src="https://img.shields.io/badge/doc-website-red">
    </a>
    <a href="https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>



<p align="center">
    <b> 简体中文 | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README.md">English</a> </b>
</p>


<h1 align="center">
    <p>DeepKE-LLM: 基于深度学习的开源中文知识图谱抽取框架 （大模型版）</p>
</h1>



# 📌 上下文学习

## 📗 单次推理

运行命令：

```bash
python run.py --mood icl
```

[配置示例](./examples/incontext_learning/icl_qwen.yaml)：

```yaml
engine: "Qwen"
model_id: "your-model-path"
temperature: 0.3
top_p: 0.9
max_tokens: 512
task: "da"
language: "ch"
in_context:
text_input: "创立"
domain:
labels: "人物, 公司"
```

参数说明：

| 参数名称    | 类型  | 意义                                                         | 限定                                                         |
| ----------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| engine      | str   | 表示所用的大模型名称。                                       | **必须**要有，且**必须**为 `MODEL_LIST `之一。               |
| api_key     | str   | 用户的API密钥。                                              | 若模型为闭源，则**必须**提供。                               |
| base_url    | str   | 基础 URL，用于指定 API 请求的基础地址。                      |                                                              |
| temperature | float | 用于控制生成文本的随机性。                                   |                                                              |
| top_p       | float | 核采样（Top-p）参数，用于控制生成文本的多样性。              |                                                              |
| max_tokens  | int   | 最大 token 数，用于限制生成文本的长度。                      |                                                              |
| stop        | str   | 停止词，用于指定生成文本时的终止条件。                       |                                                              |
| task        | str   | 参数用于指定任务类型，其中`ner`表示命名实体识别任务，`re`表示关系抽取任务`ee`表示事件抽取任务，`rte`表示三元组抽取任务。 | **必须**要有，且**必须**为 `TASK_LIST` 之一。                |
| language    | str   | 表示任务的语言。                                             | **必须**要有，且**必须**为 `LANGUAGE_LIST` 之一。            |
| in_context  | bool  | 是否为零样本设定，为`False`时表示只使用instruction提示模型进行信息抽取，为`True`时表示使用in-context的形式进行信息抽取。 |                                                              |
| instruction | str   | 规定用户自定义的提示指令，当为空时采用默认的指令。           | （不建议使用）                                               |
| data_path   | str   | 表示in-context examples的存储目录，默认为`data`文件夹。      |                                                              |
| text_input  | str   | 在命名实体识别任务(`ner`)中，`text_input`参数为预测文本；在关系抽取任务(`re`)中，`text_input`参数为文本；在事件抽取任务(`ee`)中，`text_input`参数为待预测文本；在三元组抽取任务(`rte`)中，`text_input`参数为待预测文本。 | 所有任务**必须**要有。                                       |
| domain      | str   | 在命名实体识别任务(`ner`)中，`domain`为预测文本所属领域，可为空；在关系抽取任务(`re`)中，`domain`为文本所属领域，可为空；在事件抽取任务(`ee`)中，`domain`为预测文本所属领域，可为空；在三元组抽取任务(`rte`)中，`domain`为预测文本所属领域，可为空。 | 任务为 ner、re、ee、rte 之一时，建议设置。                   |
| labels      | str   | 在命名实体识别任务(`ner`)中，`labels`为实体标签集，如无自定义的标签集，该参数可为空；在关系抽取任务(`re`)中，`labels`为关系类型标签集，如无自定义的标签集，该参数可为空。`da`中`lables`为头尾实体被预先分类的类型。 | 任务为 da 时，**必须**要有。任务为 ner、re 之一时，建议设置。多个标签时，用逗号分割。 |
| head_entity | str   | 待预测关系的头实体和尾实体。                                 | 任务为 re 时，**必须**要有。                                 |
| tail_entity | str   | 待预测关系的头实体和尾实体。                                 | 任务为 re 时，**必须**要有。                                 |
| head_type   | str   | 待预测关系的头尾实体类型。                                   | 任务为 re 时，**必须**要有。                                 |
| tail_type   | str   | 待预测关系的头尾实体类型。                                   | 任务为 re 时，**必须**要有。                                 |

## 📚 批量推理

数据集支持：

| 名称       | 下载                                                         | 数量  | 描述                                                         |
| ---------- | ------------------------------------------------------------ | ----- | ------------------------------------------------------------ |
| InstructIE | [Google drive](https://drive.google.com/file/d/1raf0h98x3GgIhaDyNn1dLle9_HvwD6wT/view?usp=sharing) [Hugging Face](https://huggingface.co/datasets/zjunlp/InstructIE) [ModelScope](https://modelscope.cn/datasets/ZJUNLP/InstructIE) [WiseModel](https://wisemodel.cn/datasets/zjunlp/InstructIE) | 30w+  | **双语**(中文和英文)基于主题的信息抽取(IE)指令数据集         |
| IEPile     | [Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) [Hugging Face](https://huggingface.co/datasets/zjunlp/iepile) [WiseModel](https://wisemodel.cn/datasets/zjunlp/IEPile) [ModelScpoe](https://modelscope.cn/datasets/ZJUNLP/IEPile) | 200w+ | 大规模(`0.32B` tokens)高质量**双语**(中文和英文)信息抽取(IE)指令微调数据集 |

数据准备同样参考微调部分的数据转换。

运行脚本：

```bash
python run.py --mood icl_batch
```

[配置参考](./examples/incontext_learning/icl_batch_chatgpt.yaml)：

```yaml
engine: "ChatGPT"
model_id: "gpt-3.5-turbo"
api_key: "your-api-key"
temperature: 0.3
top_p: 0.9
max_tokens: 512
stop: null
task: "re"
language: "ch"
in_context: false
instruction: ""
data_path: "data/ICL_Examples"
text_input: "../data/RE/try.json"
domain: ""
labels: [""]
head_entity: ""
head_type: ""
tail_entity: ""
tail_type: ""
```

参数说明同单次推理。

# 📌 微调

## 🛠️ 环境

请通过以下命令，创建并配置好环境：

```bash
cd example/llm

mkdir -p models results lora

conda create -n deepke-llm python=3.9
conda activate deepke-llm

pip install -r requirements.txt
```


## 📊 数据

现有数据集：

| 名称       | 下载                                                         | 数量  | 描述                                                         |
| ---------- | ------------------------------------------------------------ | ----- | ------------------------------------------------------------ |
| InstructIE | [1,Google drive](https://drive.google.com/file/d/1raf0h98x3GgIhaDyNn1dLle9_HvwD6wT/view?usp=sharing) <br/> [2,Hugging Face](https://huggingface.co/datasets/zjunlp/InstructIE) <br/> [3,ModelScope](https://modelscope.cn/datasets/ZJUNLP/InstructIE)<br/> [4,WiseModel](https://wisemodel.cn/datasets/zjunlp/InstructIE) | 30w+  | **双语**(中文和英文)基于主题的信息抽取(IE)指令数据集         |
| IEPile     | [1,Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) <br/> [2,Hugging Face](https://huggingface.co/datasets/zjunlp/iepile) <br/> [3,ModelScpoe](https://modelscope.cn/datasets/ZJUNLP/IEPile) <br/> [4,WiseModel](https://wisemodel.cn/datasets/zjunlp/IEPile) | 200w+ | 大规模(`0.32B` tokens)高质量**双语**(中文和英文)信息抽取(IE)指令微调数据集 |


<details>
  <summary><b>InstructIE详细信息</b></summary>
**一条数据的示例**


```json
{
  "id": "bac7c32c47fddd20966e4ece5111690c9ce3f4f798c7c9dfff7721f67d0c54a5",
  "cate": "地理地区",
  "text": "阿尔夫达尔（挪威语：Alvdal）是挪威的一个市镇，位于内陆郡，行政中心为阿尔夫达尔村。市镇面积为943平方公里，人口数量为2,424人（2018年），人口密度为每平方公里2.6人。",
  "relation": [
    {"head": "阿尔夫达尔", "head_type": "地理地区", "relation": "面积", "tail": "943平方公里", "tail_type": "度量"},
    {"head": "阿尔夫达尔", "head_type": "地理地区", "relation": "别名", "tail": "Alvdal", "tail_type": "地理地区"},
    {"head": "内陆郡", "head_type": "地理地区", "relation": "位于", "tail": "挪威", "tail_type": "地理地区"},
    {"head": "阿尔夫达尔", "head_type": "地理地区", "relation": "位于", "tail": "内陆郡", "tail_type": "地理地区"},
    {"head": "阿尔夫达尔", "head_type": "地理地区", "relation": "人口", "tail": "2,424人", "tail_type": "度量"}
  ]
}
```

字段说明:

|   字段   |                             说明                             |
| :------: | :----------------------------------------------------------: |
|    id    |                   每个数据点的唯一标识符。                   |
|   cate   |           文本的主题类别，总计12种不同的主题分类。           |
|   text   |     模型的输入文本，目标是从中抽取涉及的所有关系三元组。     |
| relation | 描述文本中包含的关系三元组，即(head, head_type, relation, tail, tail_type)。 |

转换过程请参考数据转换部分代码。

</details>


<details>
  <summary><b>IEPile详细信息</b></summary>
**一条数据的示例**

```json
{
  "task": "NER",
  "source": "MSRA",
  "instruction": "{\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"组织机构\", \"地理位置\", \"人物\"], \"input\": \"对于康有为、梁启超、谭嗣同、严复这些从旧文化营垒中走来的年轻“布衣”，他们背负着沉重的历史包袱，能够挣脱旧传统的束缚，为拯救民族的危亡而献身，实在是中华民族的脊梁。\"}",
  "output": "{\"组织机构\": [], \"地理位置\": [\"中华\"], \"人物\": [\"康有为\", \"梁启超\", \"谭嗣同\", \"严复\"]}"
}
```

字段说明:

|    字段     |                             说明                             |
| :---------: | :----------------------------------------------------------: |
|    task     | 该实例所属的任务, (`NER`、`RE`、`EE`、`EET`、`EEA`) 5种任务之一。 |
|   source    |                      该实例所属的数据集                      |
| instruction | 输入模型的指令, 经过json.dumps处理成JSON字符串, 包括`"instruction"`, `"schema"`, `"input"`三个字段 |
|   output    | 输出, 采用字典的json字符串的格式, key是schema, value是抽取出的内容 |

在`IEPile`中, **`instruction`** 的格式采纳了类JSON字符串的结构，实质上是一种字典型字符串，它由以下三个主要部分构成：
(1) **`'instruction'`**: 任务描述, 它概述了指令的执行任务(`NER`、`RE`、`EE`、`EET`、`EEA`之一)。
(2) **`'schema'`**: 待抽取的schema(`实体类型`, `关系类型`, `事件类型`)列表。
(3) **`'input'`**: 待抽取的文本。

</details>

确保结构如下：

```
DeepKE/example/llm/data
├── train.json    # 训练集
└── dev.json      # 验证集
```

数据转换：

这里需要准备**训练数据集**和**测试数据集**。

> 在使用 [convert_func.py](./instruction/convert_func.py) 脚本之前，请确保参考了 [data](./data) 目录。该目录详细说明了每种任务所需的数据格式要求。 `sample.json` 描述了转换前数据的格式，`schema.json` 展示了 schema 的组织结构， `train.json` 描述了转换后的数据格式。

> 此外，可直接使用包含12个主题（如人物、交通工具、艺术作品、自然科学、人造物品、天文对象等）的中英双语信息抽取数据集 [zjunlp/InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE)。

1. 首先, 准备训练数据集。需要将**数据格式化**以包含`instruction`、`output`字段。为此，我们提供了一个脚本 [convert_func.py](./ie2instruction/convert_func.py)，它可以将数据批量转换成模型可以直接使用的格式。

2. 其次，逐步测试数据集。在准备测试数据转换之前，请访问 [data](./data) 目录以了解各任务所需的数据结构：1）输入数据格式参见 `sample.json`；2）schema格式请查看 `schema.json`；3）转换后数据格式可参照 `train.json`。**与训练数据不同, 测试数据的输入无需包含标注字段（`entity`, `relation`, `event`）**。

使用这个命令进行数据转换（需要自己修改yaml参数调整是**训练数据集**和**测试数据集**，以及其他参数）


```
python instruction/convert_func.py
```

需要自己修改**配置或参数**：[convert.yaml](./examples/infer/convert.yaml)

参数说明：

|     字段     |                             说明                             |
| :----------: | :----------------------------------------------------------: |
|     mode     |           是用户自己选择生成训练数据还是测试数据。           |
|   src_path   |              是样例，即描述了转换前数据的格式。              |
|   tgt_path   | 是转换后的数据。**测试与训练数据不同, 测试数据的输入无需包含标注字段（`entity`, `relation`, `event`）**。 |
|   language   |     支持`zh`, `en`两种语言, 不同语言使用的指令模版不同。     |
|     task     | 目前支持['`RE`', '`NER`', '`EE`', '`EET`', '`EEA`', 'KG']任务。 |
|  split_num   | 定义单个指令中可包含的最大schema数目。默认值为4，设置为-1则不进行切分。推荐的任务切分数量依任务而异：**NER建议为6，RE、EE、EET、EEA均推荐为4、KG推荐为1**。 |
| random_sort  | 是否对指令中的schema随机排序, 默认为False, 即按字母顺序排序。 |
| split (必选) | 指定数据集类型，`train` (训练集train.json、验证集dev.json均使用`train`) 或`test`。设置 `split` 为 **test** 时，请根据任务类型选择适当的schema数量：**NER推荐为6，而RE、EE、EET、EEA推荐为4**。 |

* 转换后的训练数据将包含 `task`, `source`, `instruction`, `output` 四个字段。
* 转换后的测试数据将含有`id`, `task`, `source`, `instruction`, `label`五个字段。`label` 字段将用于后续评估。若输入数据中缺少标注字段（`entity`, `relation`, `event`），则转换后的测试数据将不包含`label`字段，适用于那些无原始标注数据的场景。

##  🧭 微调

默认方式是 Lora 微调。

下面是一些已经经过充分信息抽取指令数据训练的模型：

| 模型                      | 下载                                                         | 描述                             |
| ------------------------- | ------------------------------------------------------------ | -------------------------------- |
| llama2-13b-iepile-lora    | [huggingface](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) | （底座模型是LLaMA2-13B-Chat）    |
| baichuan2-13b-iepile-lora | [huggingface](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) | （底座模型是BaiChuan2-13B-Chat） |
| llama3-8b-iepile-lora     | [huggingface](https://huggingface.co/zjunlp/llama3-8b-iepile-lora) |                                  |
| qwen1.5-14b-iepile-lora   | [huggingface](https://huggingface.co/zjunlp/qwen1.5-14b-iepile-lora) |                                  |
| OneKE                     | [huggingface](https://huggingface.co/zjunlp/OneKE)           |                                  |

首先需要确定的是多卡还是单卡训练，检查当前GPU数量：

```bash
nvidia-smi -L
```

设置训练用卡：

```bash
CUDA_VISIBLE_DEVICES="0"  # CUDA_VISIBLE_DEVICES="0,1,2,3"
```

先创建文件夹用于存放微调结果：

```bash
output_dir='lora/llama2-13b-chat-v1'
mkdir -p ${output_dir}
```

再运行脚本：

```bash
python run.py --mood lora
```

[配置参数](./examples/infer/fine_llama.yaml)：

```bash
do_train: true
do_eval: true
overwrite_output_dir: true
model_name_or_path: 'models/llama2-13B-Chat'
stage: 'sft'
model_name: 'llama'
template: 'llama2'
train_file: 'data/train.json'
valid_file: 'data/dev.json'
output_dir: '${output_dir}'
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 4
preprocessing_num_workers: 16
num_train_epochs: 10
learning_rate: 5e-5
max_grad_norm: 0.5
optim: 'adamw_torch'
max_source_length: 400
cutoff_len: 700
max_target_length: 300
evaluation_strategy: 'epoch'
save_strategy: 'epoch'
save_total_limit: 10
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
bf16: true
bits: 4
```

参数说明：

| 字段                                                    | 描述                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| model_name                                              | 指定所需的**模型架构名称**(7B、13B、Base、Chat属于同一模型架构)。当前支持的模型包括：["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]。**请注意**，此参数应与 `--model_name_or_path` 区分。 |
| model_name_or_path                                      | 模型路径, 请到 [HuggingFace](https://huggingface.co/models) 下载相应模型。 |
| template                                                | 使用的**模板名称**，包括：`alpaca`, `baichuan`, `baichuan2`, `chatglm3`等, 请参考 [src/datamodule/template.py](./src/datamodule/template.py) 查看所有支持的模版名称, 默认使用的是`alpaca`模板, **`Chat`版本的模型建议使用配套的模版, Base版本模型可默认使用`alpaca`**。 |
| train_file, valid_file（可选）                          | 训练集和验证集的**文件路径**。注意：目前仅支持json格式的文件。`valid_file`不能指定为`test.json`文件(不包含output字段,会报错)，可以通过指定`val_set_size`参数替代`valid_file`。 |
| output_dir                                              | LoRA微调后的**权重参数保存路径**。                           |
| val_set_size                                            | **验证集的样本数量**, 默认为1000。若没有指定`valid_file`, 将会从`train_file`中划分出对应数量的样本作为验证集。 |
| per_device_train_batch_size, per_device_eval_batch_size | 每台GPU设备上的`batch_size`, 根据显存大小调整, RTX3090建议设置2~4。 |
| max_source_length, max_target_length, cutoff_len        | 最大输入、输出长度、截断长度, 截断长度可以简单地视作最大输入长度 + 最大输出长度, 需根据具体需求和显存大小设置合适值。 |
| deepspeed （可选）                                      | 使用`deepspeed`, 可设置 `--deeepspeed data/DeepSpeed/ds_config_bf16_stage2.json`。可通过设置 `bits` = 4 进行量化, RTX3090建议量化。 |

* 要了解更多关于**参数配置**的信息，请参考 [src/args](./src/args) 目录。

## ⛳ 推理

### 基础模型+Lora

以下是一些经过LoRA技术训练优化的模型(**Lora权重**)：

<details>
  <summary><b>V1版本</b></summary>

* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [knowlm-13b-ie-lora](https://huggingface.co/zjunlp/knowlm-13b-ie-lora)


| checkpoint_dir     | model_name_or_path          | moadel_name | fp16/bf16 | template |
| ------------------ | --------------------------- | ----------- | --------- | -------- |
| llama-7b-lora-ie   | llama-7b                    | llama       | fp16      | alpaca   |
| alpaca-7b-lora-ie  | alpaca-7b                   | alpaca      | fp16      | alpaca   |
| knowlm-13b-ie-lora | zjunlp/knowlm-13b-base-v1.0 | zhixi       | fp16      | alpaca   |

</details>

**V2版本(推荐)**

* [zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main)
* [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora)
* [zjunlp/llama3-8b-iepile-lora](https://huggingface.co/zjunlp/llama3-8b-iepile-lora)
* [zjunlp/qwen1.5-14b-iepile-lora](https://huggingface.co/zjunlp/qwen1.5-14b-iepile-lora)


| checkpoint_dir            | model_name_or_path | moadel_name | fp16/bf16 | template  |
| ------------------------- | ------------------ | ----------- | --------- | --------- |
| llama2-13b-iepile-lora    | LLaMA2-13B-Chat    | llama       | bf16      | llama2    |
| baichuan2-13b-iepile-lora | BaiChuan2-13B-Chat | baichuan    | bf16      | baichuan2 |
| llama3-8b-iepile-lora     | LLaMA3-8B-Instruct | llama       | bf16      | alpaca    |
| qwen1.5-14b-iepile-lora   | Qwen1.5-14B-Chat   | qwen2       | bf16      | qwen      |



要使用这些**训练好的**LoRA模型进行预测，可以执行以下命令：

```bash
python run.py --mood infer
```

[配置参考](./examples/infer/infer_llama.yaml)：

```yaml
stage: sft
model_name_or_path: 'models/llama2-13B-Chat'
checkpoint_dir: 'lora/llama2-13b-IEPile-lora'
model_name: 'llama'
template: 'llama2'
do_predict: true
input_file: 'data/input.json'
output_file: 'results/llama2-13b-IEPile-lora_output.json'
finetuning_type: 'lora'
output_dir: 'lora/test'
predict_with_generate: true
cutoff_len: 512
bf16: true
max_new_tokens: 300
bits: 4
```

参数说明：

| 字段                       | 说明                                                         |
| -------------------------- | ------------------------------------------------------------ |
| model_name_or_path         | 指定所使用的基础模型路径，必须与相应的LoRA模型匹配。         |
| checkpoint_dir             | LoRA的权重文件路径。                                         |
| output_dir                 | 此参数在推理时不起作用，可以随意指定一个路径。               |
| input_file, output_file    | 分别指定输入的测试文件路径和预测结果的输出文件路径。         |
| cutoff_len, max_new_tokens | 设置最大的输入长度和生成的新token数量，根据显存大小进行调整。 |
| bits                       | 可通过设置 `bits` = 4 进行量化, RTX3090建议量化。            |

* 在进行推理时，`model_name`, `template`, 和 `bf16` 必须与训练时的设置相同。

### IE专用模型

| checkpoint_dir | model_name_or_path | moadel_name | fp16/bf16 | template  |
| -------------- | ------------------ | ----------- | --------- | --------- |
| OneKE          | OneKE              | llama       | bf16      | llama2_zh |

**`OneKE(based on chinese-alpaca2)`** 模型下载链接：[zjunlp/OneKE](https://huggingface.co/zjunlp/OneKE)

执行以下命令：

```bash
python run.py --mood infer
```


若要使用**已训练的模型**（无LoRA或LoRA已集成到模型参数中），可以改为以下配置进行预测：

```yaml
stage: sft
model_name_or_path: 'models/OneKE'
model_name: 'llama'
template: 'llama2_zh'
do_predict: true
input_file: 'data/input.json'
output_file: 'results/OneKE_output.json'
output_dir: 'lora/test'
predict_with_generate: true
cutoff_len: 512
bf16: true
max_new_tokens: 300
bits: 4
```

`model_name_or_path`: IE专用模型权重路径

注意：对于已经量化的模型（如 4-bit 或 8-bit），不需要再调用 `.to()` 来转换设备或数据类型，因为量化模型已经预先设置好了这些信息。

### 合并基础模型+Lora导出

运行命令：

```bash
python run.py --mood export
```

将底座模型和训练的Lora权重合并, 导出模型

[配置文件](./examples/infer/export.yaml)：

```yaml
model_name_or_path: 'models/Baichuan2-13B-Chat'
checkpoint_dir: 'lora_results/baichuan2-13b-v1/checkpoint-xxx'
export_dir: 'lora_results/baichuan2-13b-v1/baichuan2-13b-v1'
stage: 'sft'
model_name: 'baichuan'
template: 'baichuan2'
output_dir: 'lora_results/test'
```

注意 `template`、`model_name` 与训练时保持一致。


### vllm加速推理

推荐环境：

```bash
pip install tiktoken
pip install peft==0.7.1
pip install transformers==4.41.2

pip install vllm==0.3.0
pip install jinja2==3.0.1
pip install pydantic==1.9.2

ip route add 8.8.8.8 via 127.0.0.1
```

运行脚本

```bash
 python run.py --mood infer_vllm
```

[参考配置 vllm_baichuan.yaml](./examples/infer/infer_baichuan_vllm.yaml):

```yaml
stage: sft
model_name_or_path: 'lora_results/baichuan2-13b-v1/baichuan2-13b-v1'
model_name: 'baichuan'
template: 'baichuan2'
do_predict: true
input_file: 'data/input.json'
output_file: 'results/baichuan2-13b-IEPile-lora_output.json'
output_dir: 'lora_results/test'
batch_size: 4
predict_with_generate: true
max_source_length: 1024
bf16: true
max_new_tokens: 512
```




## 🧾 评估

我们提供了评估各个任务F1分数的脚本。

```bash
python run.py -m evaluate
```

[配置文件](./examples/infer/evaluate.yaml)：

```bash
path1: "results/baichuan2-13b-iepile-lora_output.json"
task: "RE"
language: "zh"
sort_by: "source"
match_mode: "normal"
metrics_list: "f1"
```

参数说明：

* `path1` 是模型的输出文件, 其中一条数据样本如下所示, 经测试数据转换脚本转换后的数据(`test.json`)具有`id`、`instruction`、`label`字段, `output`字段是经过模型预测脚本后的模型真实输出。

* `task`: 目前支持['RE', 'NER', 'EE', 'EET', 'EEA']五类任务。
* 可以设置 `sort_by` 为 `source`, 分别计算每个数据集上的F1分数。

-------

✨
