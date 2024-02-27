<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="assets/LLM_logo.png" width="400"/></a>
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
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md">简体中文</a> </b>
</p>

<h1 align="center">
    <p>DeepKE-LLM: A Large Language Model Based<br>Knowledge Extraction Toolkit</p>
</h1>


- [环境依赖](#环境依赖)
- [新闻](#新闻)
- [数据](#数据)
- [模型](#模型)
  - [LLaMA系列](#llama系列)
    - [案例一：LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调llama完成ccks2023指令驱动的知识图谱构建-英文--中文)
    - [案例二：使用智析完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例二使用智析完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [ChatGLM](#chatglm)
    - [案例一：LoRA微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调chatglm完成ccks2023指令驱动的知识图谱构建-英文--中文)
    - [案例二：P-Tuning微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例二p-tuning微调chatglm完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [MOSS](#moss)
    - [案例一：Lora微调Moss完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调moss完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [Baichuan](#baichuan)
    - [Case 1: Lora微调Baichuan完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#case-1-lora微调baichuan完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [CPM-Bee](#cpm-bee)
    - [案例一：OpenDelta微调CPM-Bee完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一opendelta微调cpm-bee完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [GPT系列](#gpt系列)
    - [案例一：ICL使用大语言模型进行信息抽取 英文 | 中文](#案例一icl使用大语言模型进行信息抽取-英文--中文)
    - [案例二：ICL使用大语言模型进行数据增强 英文 | 中文](#案例二icl使用大语言模型进行数据增强-英文--中文)
    - [案例三：ICL使用大语言模型完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例三icl使用大语言模型完成ccks2023指令驱动的知识图谱构建-英文--中文)
    - [案例四：ICL释放大型语言模型进行少样本关系抽取中的能力 英文 | 中文](#案例四icl释放大型语言模型进行少样本关系抽取中的能力-英文--中文)
    - [案例五：CodeKGC-基于代码语言模型的知识图谱构建 英文 | 中文](#案例五codekgc-基于代码语言模型的知识图谱构建-英文--中文)
- [方法](#方法)
  - [方法一：In-Context Learning(ICL)](#方法一in-context-learningicl)
  - [方法二：LoRA](#方法二lora)
  - [方法三：P-Tuning](#方法三p-tuning)
- [引用](#引用)

---

# 环境依赖

大模型时代, DeepKE-LLM采用全新的环境依赖
```
conda create -n deepke-llm python=3.9
conda activate deepke-llm
cd example/llm
pip install -r requirements.txt
```
注意！！是example/llm文件夹下的 `requirements.txt`


# 新闻

* [2024/02] 我们发布了一个大规模(`0.32B` tokens)高质量**双语**(中文和英文)信息抽取(IE)指令微调数据集，名为 [IEPile](https://huggingface.co/datasets/zjunlp/iepie), 以及基于 `IEPile` 训练的两个模型[baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora)、[llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora)。
* [2023/11] [knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie/tree/main)权重进行了更新，这次更新主要是调整了NAN输出，缩短了推理长度，并支持了不指定schema的指令。
* [2023/10] 我们发布了一个新的**双语**(中文和英文)基于主题的信息抽取(IE)指令数据集，名为[InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE)和[论文](https://arxiv.org/abs/2305.11527)。
* [2023/08] 推出了专门用于信息抽取(IE)的KnowLM版本，命名为[knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie/tree/main)。
* [2023/07] 发布了训练所使用的部分指令数据集，包括[knowlm-ke](https://huggingface.co/datasets/zjunlp/knowlm-ke)和[KnowLM-IE](https://huggingface.co/datasets/zjunlp/KnowLM-IE)。
* [2023/06] 发布了第一版的预训练权重[knowlm-13b-base-v1.0](https://huggingface.co/zjunlp/knowlm-13b-base-v1.0)和第一版的[zhixi-13b-lora](https://huggingface.co/zjunlp/zhixi-13b-lora)。
* [2023/05] 我们启动了基于指令的信息抽取项目。


# 数据

**现有数据集**

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



# 模型

## LLaMA系列
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/llama.jpg" alt="LLaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>

LLaMA，它是一组从7B到65B参数的基础语言模型。我们还提供了基于[KnowLM](https://github.com/zjunlp/KnowLM)框架的抽取大模型`智析`的支持。其首先（1）使用中英双语语料首先对LLaMA（13B）进行进一步全量预训练，在尽可能保留原来的英文和代码能力的前提下，进一步提高模型对于中文理解能力和知识储备；接着（2）使用指令数据集对第一步的模型微调，来提高语言模型对于人类指令的理解。模型详细信息请参考[KnowLM](https://github.com/zjunlp/KnowLM).

### 案例一：LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md/#42lora-fine-tuning-with-llama) | [中文](./InstructKGC/README_CN.md/#42lora微调llama)

### 案例二：使用智析完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md/#44lora-fine-tuning-with-zhixi-智析) | [中文](./InstructKGC/README_CN.md/#44lora微调智析)



## ChatGLM
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatglm_logo.png" alt="ChatGLM" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### 案例一：LoRA微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC//README.md/#46lora-fine-tuning-with-chatglm) | [中文](./InstructKGC//README_CN.md/#46lora微调chatglm) 
### 案例二：P-Tuning微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md/#51p-tuning-fine-tuning-with-chatglm) | [中文](./InstructKGC/README_CN.md/#51p-tuning微调chatglm)




## MOSS
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/moss_logo.png" alt="ChatGLM" style="width: 25%; min-width: 25px; display: block; margin: auto;"></a>

### 案例一：Lora微调Moss完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC//README.md/#47lora-fine-tuning-with-moss) | [中文](./InstructKGC//README_CN.md/#47lora微调moss) 




## Baichuan
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/baichuan_logo.png" alt="Baichuan" style="width: 25%; min-width: 25px; display: block; margin: auto;"></a>


### Case 1: Lora微调Baichuan完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC//README.md/#48lora-fine-tuning-with-baichuan) | [中文](./InstructKGC//README_CN.md/#48lora微调baichuan) 




## CPM-Bee
### 案例一：OpenDelta微调CPM-Bee完成CCKS2023指令驱动的知识图谱构建 [英文](./CPM-Bee/README.md) | [中文](./CPM-Bee/README_CN.md) 




## GPT系列

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatgpt_logo.png" alt="GPT" style="width: 18%; min-width: 18px; display: block; margin: auto;"></a>
</p>

### 案例一：ICL使用大语言模型进行信息抽取 [英文](./LLMICL/README.md/#ie-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行信息抽取)


### 案例二：ICL使用大语言模型进行数据增强 [英文](./LLMICL/README.md/#data-augmentation-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行数据增强)


### 案例三：ICL使用大语言模型完成CCKS2023指令驱动的知识图谱构建 [英文](./LLMICL/README.md/#ccks2023-instruction-based-knowledge-graph-construction-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建)

### 案例四：ICL释放大型语言模型进行少样本关系抽取中的能力 [英文](./UnleashLLMRE/README.md) | [中文](./UnleashLLMRE/README_CN.md)

### 案例五：CodeKGC-基于代码语言模型的知识图谱构建 [英文](./CodeKGC/README.md) | [中文](./CodeKGC/README_CN.md)

为了更好地处理知识图谱构建中的关系三元组抽取（RTE）任务，我们设计了代码形式的提示建模关系三元组的结构，并使用代码语言模型（Code-LLM）生成更准确的预测。代码形式提示构建的关键步骤是将（文本，输出三元组）对转换成Python中的语义等价的程序语言。




# 方法

## 方法一：In-Context Learning(ICL)
[In-Context Learning](http://arxiv.org/abs/2301.00234) 是一种指导大语言模型的方法，可以提升其在特定任务上的表现。它通过在特定上下文中进行迭代学习，对模型进行微调和训练，以使其更好地理解和应对特定领域的需求。通过 `In-Context Learning`，我们可以让大语言模型具备信息抽取、数据增强以及指令驱动的知识图谱构建等功能。

## 方法二：LoRA
LoRA通过学习秩分解矩阵对来减少可训练参数的数量，同时冻结原始权重。这大大降低了适用于特定任务的大型语言模型的存储需求，并在部署期间实现了高效的任务切换，而无需引入推理延迟。详细可查看原论文[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

## 方法三：P-Tuning
PT方法，即P-Tuning方法，参考[ChatGLM官方代码](https://link.zhihu.com/?target=https%3A//github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md) ，是一种针对于大模型的soft-prompt方法。
[P-Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.10385)，仅对大模型的Embedding加入新的参数。
[P-Tuning-V2](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2110.07602)，将大模型的Embedding和每一层前都加上新的参数。




# 引用

如果您使用了本项目代码， 烦请引用下列论文: 

```bibtex
@misc{knowlm,
  author = {Ningyu Zhang and Jintian Zhang and Xiaohan Wang and Honghao Gui and Kangwei Liu and Yinuo Jiang and Xiang Chen and Shengyu Mao and Shuofei Qiao and Yuqi Zhu and Zhen Bi and Jing Chen and Xiaozhuan Liang and Yixin Ou and Runnan Fang and Zekun Xi and Xin Xu and Lei Li and Peng Wang and Mengru Wang and Yunzhi Yao and Bozhong Tian and Yin Fang and Guozhou Zheng and Huajun Chen},
  title = {KnowLM Technical Report},
  year = {2023},
 url = {http://knowlm.zjukg.cn/},
}
```
