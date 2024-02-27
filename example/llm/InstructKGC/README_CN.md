# InstructionKGC-指令驱动的自适应知识图谱构建

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README.md">English</a> | 简体中文 </b>
</p>


- [InstructionKGC-指令驱动的自适应知识图谱构建](#instructionkgc-指令驱动的自适应知识图谱构建)
  - [新闻](#新闻)
  - [🎯 1.任务目标](#-1任务目标)
  - [📊 2.数据](#-2数据)
    - [2.1现有数据集](#21现有数据集)
    - [2.2训练数据转换](#22训练数据转换)
    - [2.3测试数据转换](#23测试数据转换)
  - [🚴 3.准备](#-3准备)
    - [🛠️ 3.1环境](#️-31环境)
    - [🐐 3.2模型](#-32模型)
  - [🌰 4.LoRA微调](#-4lora微调)
    - [4.1基础参数](#41基础参数)
    - [4.2LoRA微调LLaMA](#42lora微调llama)
    - [4.3LoRA微调Alpaca](#43lora微调alpaca)
    - [4.4LoRA微调智析](#44lora微调智析)
    - [4.5LoRA微调Vicuna](#45lora微调vicuna)
    - [4.6LoRA微调ChatGLM](#46lora微调chatglm)
    - [4.7LoRA微调Moss](#47lora微调moss)
    - [4.8LoRA微调Baichuan](#48lora微调baichuan)
    - [4.9领域内数据继续训练](#49领域内数据继续训练)
  - [🥊 5.P-Tuning微调](#-5p-tuning微调)
    - [5.1P-Tuning微调ChatGLM](#51p-tuning微调chatglm)
  - [🔴 6.预测](#-6预测)
    - [6.1LoRA预测](#61lora预测)
      - [6.1.1基础模型+Lora](#611基础模型lora)
      - [6.1.2IE专用模型](#612ie专用模型)
    - [6.2P-Tuning预测](#62p-tuning预测)
  - [🧾 7.评估](#-7评估)
  - [👋 8.Acknowledgment](#-8acknowledgment)
  - [9.引用](#9引用)


## 新闻
* [2024/02] 我们发布了一个大规模(`0.32B` tokens)高质量**双语**(中文和英文)信息抽取(IE)指令微调数据集，名为 [IEPile](https://huggingface.co/datasets/zjunlp/iepie), 以及基于 `IEPile` 训练的两个模型[baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora)、[llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora)。
* [2023/10] 我们发布了一个新的**双语**(中文和英文)基于主题的信息抽取(IE)指令数据集，名为[InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE)和[论文](https://arxiv.org/abs/2305.11527)。
* [2023/08] 我们推出了专用于信息抽取(IE)的13B模型，名为[knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie/tree/main)。
* [2023/05] 我们启动了基于指令的信息抽取项目。



## 🎯 1.任务目标

我们将`Instruction-based KGC`制定为一种遵循指令的自回归生成任务。模型首先需要理解指令识别其意图，然后根据指令内容，模型会基于输入的文本抽取相应的三元组并以指定的格式输出。本文的 **`instruction`** 格式采纳了类JSON字符串的结构，实质上是一种字典型字符串。它由以下三个字段构成：
(1) **`'instruction'`**，即任务描述，以自然语言指定模型扮演的角色以及需要完成的任务；
(2) **`'schema'`**，这是一份需提取的标签列表，明确指出了待抽取信息的关键字段，反应用户的需求，是动态可变的；
(3) **`'input'`**，指的是用于信息抽取的源文本。各类任务对应的指令样例。


以下是一条**数据实例**：

```json
{
  "instruction": "{\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"组织机构\", \"地理位置\", \"人物\"], \"input\": \"对于康有为、梁启超、谭嗣同、严复这些从旧文化营垒中走来的年轻“布衣”，他们背负着沉重的历史包袱，能够挣脱旧传统的束缚，为拯救民族的危亡而献身，实在是中华民族的脊梁。\"}", 
  "output": "{\"组织机构\": [], \"地理位置\": [\"中华\"], \"人物\": [\"康有为\", \"梁启超\", \"谭嗣同\", \"严复\"]}"
}
```

待抽取的schema列表是 ["组织机构", "地理位置", "人物"], 待抽取的文本是"*对于康有为、梁启超、谭嗣同、严复这些从旧文化营垒中走来的年轻“布衣”，他们背负着沉重的历史包袱，能够挣脱旧传统的束缚，为拯救民族的危亡而献身，实在是中华民族的脊梁。*", 输出是 `{"组织机构": [], "地理位置": ["中华"], "人物": ["康有为", "梁启超", "谭嗣同", "严复"]}`

> 注意输出中的 schema 顺序与 instruction 中的 schema 顺序一致


<details>
  <summary><b>更多任务的数据实例</b></summary>

```json
{
  "instruction": "{\"instruction\": \"你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"国籍\", \"作者\", \"毕业院校\", \"主角\"], \"input\": \"对比日本动画电影在中日两国的票房表现，可以发现，日漫风格的动画，在国内也有圈层限制，即便是宫崎骏《千与千寻》、新海诚《你的名字》，这类日本动画票房榜首的电影，国内票房也停留在5亿左右\"}", 
  "output": "{\"国籍\": [], \"作者\": [{\"subject\": \"你的名字\", \"object\": \"新海诚\"}], \"毕业院校\": [], \"主角\": []}"
}

{
  "instruction": "{\"instruction\": \"你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件，不存在的事件返回空列表，不存在的论元返回NAN，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。\", \"schema\": [{\"event_type\": \"人生-求婚\", \"trigger\": true, \"arguments\": [\"求婚对象\"]}, {\"event_type\": \"人生-订婚\", \"trigger\": true, \"arguments\": [\"订婚主体\", \"时间\"]}, {\"event_type\": \"灾害/意外-坍/垮塌\", \"trigger\": true, \"arguments\": [\"受伤人数\", \"坍塌主体\"]}, {\"event_type\": \"人生-失联\", \"trigger\": true, \"arguments\": [\"地点\", \"失联者\"]}], \"input\": \"郭碧婷订婚后，填资料依旧想要填单身，有谁注意向佐说了什么？\"}", 
  "output": "{\"人生-求婚\": [], \"人生-订婚\": [{\"trigger\": \"订婚\", \"arguments\": {\"订婚主体\": [\"向佐\", \"郭碧婷\"], \"时间\": \"NAN\"}}], \"灾害/意外-坍/垮塌\": [], \"人生-失联\": []}"
}
```

</details>


[instruction.py](./ie2instruction/convert/utils/instruction.py) 中提供了各个任务的指令模版。



> **注意**⚠️: 老版的数据样式请参考[kg2instruction/README.md](./kg2instruction/README.md)



## 📊 2.数据


### 2.1现有数据集

| 名称 | 下载 | 数量 | 描述 |
| --- | --- | --- | --- |
| InstructIE | [Google drive](https://drive.google.com/file/d/1raf0h98x3GgIhaDyNn1dLle9_HvwD6wT/view?usp=sharing) <br/> [Hugging Face](https://huggingface.co/datasets/zjunlp/InstructIE) <br/> [ModelScope](https://modelscope.cn/datasets/ZJUNLP/InstructIE)<br/> [WiseModel](https://wisemodel.cn/datasets/zjunlp/InstructIE) | 30w+ | **双语**(中文和英文)基于主题的信息抽取(IE)指令数据集 |
| IEPile | [Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) <br/> [Hugging Face](https://huggingface.co/datasets/zjunlp/iepile) <br/> [WiseModel](https://wisemodel.cn/datasets/zjunlp/IEPile) <br/> [ModelScpoe](https://modelscope.cn/datasets/ZJUNLP/IEPile) | 200w+ | 大规模(`0.32B` tokens)高质量**双语**(中文和英文)信息抽取(IE)指令微调数据集 |


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

各字段的说明:

|    字段      |                             说明                             |
| :---------: | :----------------------------------------------------------: |
|     id      |                       每个数据点的唯一标识符。                       |
|    cate     |           文本的主题类别，总计12种不同的主题分类。               |
|    text     | 模型的输入文本，目标是从中抽取涉及的所有关系三元组。                  |
|  relation   |   描述文本中包含的关系三元组，即(head, head_type, relation, tail, tail_type)。   |

需要参考数据转换

</details>


<details>
  <summary><b>IEPile详细信息</b></summary>


`IEPile` 中的每条数据均包含 `task`, `source`, `instruction`, `output` 4个字段, 以下是各字段的说明

| 字段 | 说明 |
| :---: | :---: |
| task | 该实例所属的任务, (`NER`、`RE`、`EE`、`EET`、`EEA`) 5种任务之一。 |
| source | 该实例所属的数据集 |
| instruction | 输入模型的指令, 经过json.dumps处理成JSON字符串, 包括`"instruction"`, `"schema"`, `"input"`三个字段 |
| output | 输出, 采用字典的json字符串的格式, key是schema, value是抽取出的内容 |


在`IEPile`中, **`instruction`** 的格式采纳了类JSON字符串的结构，实质上是一种字典型字符串，它由以下三个主要部分构成：
(1) **`'instruction'`**: 任务描述, 它概述了指令的执行任务(`NER`、`RE`、`EE`、`EET`、`EEA`之一)。
(2) **`'schema'`**: 待抽取的schema(`实体类型`, `关系类型`, `事件类型`)列表。
(3) **`'input'`**: 待抽取的文本。


以下是一条**数据实例**：

```json
{
  "task": "NER", 
  "source": "MSRA", 
  "instruction": "{\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"组织机构\", \"地理位置\", \"人物\"], \"input\": \"对于康有为、梁启超、谭嗣同、严复这些从旧文化营垒中走来的年轻“布衣”，他们背负着沉重的历史包袱，能够挣脱旧传统的束缚，为拯救民族的危亡而献身，实在是中华民族的脊梁。\"}", 
  "output": "{\"组织机构\": [], \"地理位置\": [\"中华\"], \"人物\": [\"康有为\", \"梁启超\", \"谭嗣同\", \"严复\"]}"
}
```

该数据实例所属任务是 `NER`, 所属数据集是 `MSRA`, 待抽取的schema列表是 ["组织机构", "地理位置", "人物"], 待抽取的文本是"*对于康有为、梁启超、谭嗣同、严复这些从旧文化营垒中走来的年轻“布衣”，他们背负着沉重的历史包袱，能够挣脱旧传统的束缚，为拯救民族的危亡而献身，实在是中华民族的脊梁。*", 输出是 `{"组织机构": [], "地理位置": ["中华"], "人物": ["康有为", "梁启超", "谭嗣同", "严复"]}`

</details>


### 2.2训练数据转换

首先, 需要将**数据格式化**以包含`instruction`、`output`字段。为此，我们提供了一个脚本 [convert_func.py](./ie2instruction/convert_func.py)，它可以将数据批量转换成模型可以直接使用的格式。

> 在使用 [convert_func.py](./ie2instruction/convert_func.py) 脚本之前，请确保参考了 [data](./data) 目录。该目录详细说明了每种任务所需的数据格式要求。 `sample.json` 描述了转换前数据的格式，`schema.json` 展示了 schema 的组织结构， `train.json` 描述了转换后的数据格式。

> 此外，可直接使用包含12个主题（如人物、交通工具、艺术作品、自然科学、人造物品、天文对象等）的中英双语信息抽取数据集 [zjunlp/InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE)。


```bash
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/train.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NER \
    --split_num 6 \       
    --random_sort \
    --split train
```

* `language`: 支持`zh`, `en`两种语言, 不同语言使用的指令模版不同。
* `task`: 目前支持['`RE`', '`NER`', '`EE`', '`EET`', '`EEA`']五类任务。
* `split_num`: 定义单个指令中可包含的最大schema数目。默认值为4，设置为-1则不进行切分。推荐的任务切分数量依任务而异：**NER建议为6，RE、EE、EET、EEA均推荐为4**。
* `random_sort`: 是否对指令中的schema随机排序, 默认为False, 即按字母顺序排序。
* `split`: 指定数据集类型，可选`train`或`test`。

转换后的训练数据将包含 `task`, `source`, `instruction`, `output` 四个字段。


### 2.3测试数据转换

在准备测试数据转换之前，请访问 [data](./data) 目录以了解各任务所需的数据结构：1）输入数据格式参见 `sample.json`；2）schema格式请查看 `schema.json`；3）转换后数据格式可参照 `train.json`。**与训练数据不同, 测试数据的输入无需包含标注字段（`entity`, `relation`, `event`）**。


```bash
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/test.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NER \
    --split_num 6 \
    --split test
```

设置 `split` 为 **test** 时，请根据任务类型选择适当的schema数量：**NER推荐为6，而RE、EE、EET、EEA推荐为4**。转换后的测试数据将含有`id`, `task`, `source`, `instruction`, `label`五个字段。

`label` 字段将用于后续评估。若输入数据中缺少标注字段（`entity`, `relation`, `event`），则转换后的测试数据将不包含`label`字段，适用于那些无原始标注数据的场景。





## 🚴 3.准备


### 🛠️ 3.1环境
在开始之前，请确保根据[DeepKE/example/llm/README_CN.md](../README_CN.md/#环境依赖)中的指导创建了适当的Python虚拟环境。创建并配置好**虚拟环境**后，请通过以下命令激活名为 `deepke-llm` 的环境：

```bash
conda activate deepke-llm
```


```bash
mkdir results
mkdir lora
mkdir data
```

数据放在目录 `./data` 中。


### 🐐 3.2模型

以下是本仓库代码支持的一些基础模型：[[llama](https://huggingface.co/meta-llama), [alpaca](https://github.com/tloen/alpaca-lora), [vicuna](https://huggingface.co/lmsys), [zhixi](https://github.com/zjunlp/KnowLM), [falcon](https://huggingface.co/tiiuae), [baichuan](https://huggingface.co/baichuan-inc), [chatglm](https://huggingface.co/THUDM), [qwen](https://huggingface.co/Qwen), [moss](https://huggingface.co/fnlp), [openba](https://huggingface.co/OpenBA)]



## 🌰 4.LoRA微调

下面是一些已经经过充分信息抽取指令数据训练的模型：

* [zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) （底座模型是LLaMA2-13B-Chat）
* [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) （底座模型是BaiChuan2-13B-Chat）
* [zjunlp/knowlm-ie-v2](https://huggingface.co/zjunlp/knowlm-ie-v2)


### 4.1基础参数

> 重要提示：以下的所有命令均应在InstrctKGC目录下执行。例如，如果您想运行微调脚本，您应该使用如下命令：bash ft_scripts/fine_llama.bash。请确保您的当前工作目录正确。


```bash
output_dir='lora/llama2-13b-chat-v1'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1287 src/finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/llama2-13b-chat' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama2' \
    --train_file 'data/train.json' \
    --valid_file 'data/dev.json' \
    --output_dir=${output_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --optim "adamw_torch" \
    --max_source_length 400 \
    --cutoff_len 700 \
    --max_target_length 300 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 \
    --deepspeed configs/ds_config_bf16.json
```

* `model_name`: 指定所需的**模型架构名称**(7B、13B、Base、Chat属于同一模型架构)。当前支持的模型包括：["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]。**请注意**，此参数应与 `--model_name_or_path` 区分。
* `model_name_or_path`: 模型路径, 请到 [HuggingFace](https://huggingface.co/models) 下载相应模型。
* `template`: 使用的**模板名称**，包括：`alpaca`, `baichuan`, `baichuan2`, `chatglm3`等, 请参考 [src/datamodule/template.py](./src/datamodule/template.py) 查看所有支持的模版名称, 默认使用的是`alpaca`模板, **`Chat`版本的模型建议使用配套的模版, Base版本模型可默认使用`alpaca`**。
* `train_file`, `valid_file（可选）`: 训练集和验证集的**文件路径**。注意：目前仅支持json格式的文件。
* `output_dir`: LoRA微调后的**权重参数保存路径**。
* `val_set_size`: **验证集的样本数量**, 默认为1000。
* `per_device_train_batch_size`, `per_device_eval_batch_size`: 每台GPU设备上的`batch_size`, 根据显存大小调整, RTX3090建议设置2~4。
* `max_source_length`, `max_target_length`, `cutoff_len`: 最大输入、输出长度、截断长度, 截断长度可以简单地视作最大输入长度 + 最大输出长度, 需根据具体需求和显存大小设置合适值。
* `deepspeed`: 设备资源不够可去掉。

> 可通过设置 `bits` = 4 进行量化, RTX3090建议量化。

* 要了解更多关于**参数配置**的信息，请参考 [src/utils/args](./src/args) 目录。


### 4.2LoRA微调LLaMA

微调LLaMA模型的具体脚本可以在 [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash) 中找到。


### 4.3LoRA微调Alpaca

微调Alpaca模型时，您可遵循与[微调LLaMA模型](./README_CN.md/#42lora微调llama)类似的步骤。要进行微调，请对[ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash)文件做出以下**修改**：

```bash
output_dir='path to save Alpaca Lora'
--model_name_or_path 'path or name to Alpaca' \
--template 'alpaca' \
--model_name 'alpaca' \
```

1. 对于template，我们**默认使用alpaca模板**。
2. `model_name = alpaca`


### 4.4LoRA微调智析

```bash
output_dir='path to save Zhixi Lora'
--model_name_or_path 'path or name to Zhixi' \
--model_name 'zhixi' \
--template 'alpaca' \
```

1. 由于Zhixi目前只有13b的模型, 建议相应地减小批处理大小batch size
2. 对于template，我们**默认使用alpaca模板**。
3. `model_name = zhixi`




### 4.5LoRA微调Vicuna

相应的脚本在 [ft_scripts/fine_vicuna.bash](./ft_scripts//fine_vicuna.bash)

1. 由于Vicuna-7b-delta-v1.1所使用的template与`alpaca`**模版不同**, 因此需要设置 `template vicuna`。
2. `model_name = vicuna`



### 4.6LoRA微调ChatGLM

相应的脚本在 [ft_scripts/fine_chatglm.bash](./ft_scripts//fine_chatglm.bash)

1. ChatGLM模型我们采用[THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)
2. `model_name = chatglm`
3. `template chatglm3`



### 4.7LoRA微调Moss

相应的脚本在 [ft_scripts/fine_moss.bash](./ft_scripts/fine_moss.bash)

1. Moss模型我们采用[moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
2. `model_name = moss`
  

### 4.8LoRA微调Baichuan

相应的脚本在 [ft_scripts/fine_baichuan.bash](./ft_scripts/fine_baichuan.bash)

1. Baichuan模型我们采用[baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
2. **请确保torch版本保持在2.0.0, 否则可能出现问题**
3. `model_name = baichuan`
4. `template baichuan2`
5. 我们建议使用 `--bf16`
6. 如果出现在eval后保存时爆显存请设置 `evaluation_strategy no`


### 4.9领域内数据继续训练

尽管 `llama2-13b-iepile-lora`、`baichuan2-13b-iepile-lora` 等模型已在多个通用数据集上接受了广泛的指令微调，并因此获得了一定的**通用信息抽取能力**，但它们在**特定领域**（如`法律`、`教育`、`科学`、`电信`）的数据处理上可能仍显示出一定的局限性。针对这一挑战，建议对这些模型在特定领域的数据集上进行**二次训练**。这将有助于模型更好地适应特定领域的语义和结构特征，从而增强其在**该领域内的信息抽取能力**。


```bash
output_dir='lora/llama2-13b-chat-v1-continue'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1287 src/finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/llama2-13B-Chat' \
    --checkpoint_dir 'lora/llama2-13b-iepile-lora' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama2' \
    --train_file 'data/train.json' \
    --valid_file 'data/dev.json' \
    --output_dir=${output_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --optim "adamw_torch" \
    --max_source_length 400 \
    --cutoff_len 700 \
    --max_target_length 300 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 
```

* 若要基于微调后的LoRA权重继续训练，仅需将 `checkpoint_dir` 参数指向LoRA权重路径，例如设置为`'zjunlp/llama2-13b-iepile-lora'`。

> 可通过设置 `bits` = 4 进行量化, RTX3090建议量化。

> 请注意，在使用 `llama2-13b-iepile-lora`、`baichuan2-13b-iepile-lora` 时，保持lora_r和lora_alpha均为64，对于这些参数，我们不提供推荐设置。

* 若要基于微调后的模型权重继续训练，只需设定 `model_name_or_path` 参数为权重路径，如`'zjunlp/KnowLM-IE-v2'`，无需设置`checkpoint_dir`。


脚本可以在 [ft_scripts/fine_continue.bash](./ft_scripts/fine_continue.bash) 中找到。



## 🥊 5.P-Tuning微调

### 5.1P-Tuning微调ChatGLM

你可以通过下面的命令使用P-Tuning方法来finetune模型:

```bash
deepspeed --include localhost:0 src/finetuning_pt.py \
  --train_path data/train.json \
  --model_dir /model \
  --num_train_epochs 20 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --output_dir output_dir_pt \
  --log_steps 10 \
  --max_len 768 \
  --max_src_len 450 \
  --pre_seq_len 16 \
  --prefix_projection true
```


## 🔴 6.预测

### 6.1LoRA预测

#### 6.1.1基础模型+Lora

以下是一些经过LoRA技术训练优化的模型(**Lora权重**)：

<details>
  <summary><b>V1版本</b></summary>

* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [knowlm-13b-ie-lora](https://huggingface.co/zjunlp/knowlm-13b-ie-lora)


| checkpoint_dir | model_name_or_path | moadel_name | fp16/bf16 | template | 
| --- | --- | --- | --- | --- |
| llama-7b-lora-ie | llama-7b | llama | fp16 | alpaca |
| alpaca-7b-lora-ie | alpaca-7b | alpaca | fp16 | alpaca |
| knowlm-13b-ie-lora | zhixi | fp16 | alpaca |

</details>

<details>
  <summary><b>V2版本(推荐)</b></summary>

* [zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) 
* [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) 
* [zjunlp/knowlm-ie-v2](https://huggingface.co/zjunlp/knowlm-ie-v2)


| checkpoint_dir | model_name_or_path | moadel_name | fp16/bf16 | template | 
| --- | --- | --- | --- | --- |
| llama2-13b-iepile-lora | LLaMA2-13B-Chat | llama | bf16 | llama2 |
| baichuan2-13b-iepile-lora | BaiChuan2-13B-Chat | baichuan | bf16 | baichuan2 |

</details>


要使用这些**训练好的**LoRA模型进行预测，可以执行以下命令：

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'models/llama2-13B-Chat' \
    --checkpoint_dir 'lora/llama2-13b-IEPile-lora' \
    --model_name 'llama' \
    --template 'llama2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/llama2-13b-IEPile-lora_output.json' \
    --finetuning_type lora \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --cutoff_len 512 \
    --bf16 \
    --max_new_tokens 300
```

* 在进行推理时，`model_name`, `template`, 和 `bf16` 必须与训练时的设置相同。
* `model_name_or_path`: 指定所使用的基础模型路径，必须与相应的LoRA模型匹配。
* `checkpoint_dir`: LoRA的权重文件路径。
* `output_dir`: 此参数在推理时不起作用，可以随意指定一个路径。
* `input_file`, `output_file`: 分别指定输入的测试文件路径和预测结果的输出文件路径。
* `cutoff_len`, `max_new_tokens`: 设置最大的输入长度和生成的新token数量，根据显存大小进行调整。

> 可通过设置 `bits` = 4 进行量化, RTX3090建议量化。


#### 6.1.2IE专用模型
若要使用**已训练的模型**（无LoRA或LoRA已集成到模型参数中），可以执行以下命令进行预测：

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'models/KnowLM-IE-v2' \
    --model_name 'baichuan' \
    --template 'baichuan2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/KnowLM-IE-v2_output.json' \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --cutoff_len 512 \
    --bf16 \
    --max_new_tokens 300 
```

`model_name_or_path`: IE专用模型权重路径



### 6.2P-Tuning预测

你可以通过下面的命令使用训练好的P-Tuning模型在比赛测试集上预测输出:
```bash
CUDA_VISIBLE_DEVICES=0 python src/inference_pt.py \
  --test_path data/valid.json \
  --device 0 \
  --ori_model_dir /model \
  --model_dir /output_dir_lora/global_step- \
  --max_len 768 \
  --max_src_len 450
```


## 🧾 7.评估

我们提供了评估各个任务F1分数的脚本。

```bash
python ie2instruction/eval_func.py \
  --path1 data/NER/processed.json \
  --task NER 
```

* `task`: 目前支持['RE', 'NER', 'EE', 'EET', 'EEA']五类任务。
* 可以设置 `sort_by` 为 `source`, 分别计算每个数据集上的F1分数。




## 👋 8.Acknowledgment

Part of the code is derived from [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We extend our gratitude for their contributions!


## 9.引用
如果您使用IEPile或代码，请引用以下论文：

```bibtex
@article{DBLP:journals/corr/abs-2305-11527,
  author       = {Honghao Gui and Shuofei Qiao and Jintian Zhang and Hongbin Ye and Mengshu Sun and Lei Liang and Huajun Chen and Ningyu Zhang},
  title        = {InstructIE: A Bilingual Instruction-based Information Extraction Dataset},
  journal      = {CoRR},
  volume       = {abs/2305.11527},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.11527},
  doi          = {10.48550/arXiv.2305.11527},
  eprinttype    = {arXiv},
  eprint       = {2305.11527},
  timestamp    = {Thu, 25 May 2023 15:41:47 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-11527.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-2402-14710,
  author       = {Honghao Gui and
                  Hongbin Ye and
                  Lin Yuan and
                  Ningyu Zhang and
                  Mengshu Sun and
                  Lei Liang and
                  Huajun Chen},
  title        = {IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus},
  journal      = {CoRR},
  volume       = {abs/2402.14710},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2402.14710},
  doi          = {10.48550/ARXIV.2402.14710},
  eprinttype   = {arXiv},
  eprint       = {2402.14710},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2402-14710.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```