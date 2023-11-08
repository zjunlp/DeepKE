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


- [Requirements](#requirements)
- [Data](#data)
- [Models](#models)
  - [LLaMA-series](#llama-series)
    - [Case 1: LoRA Fine-tuning of LLaMA for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-lora-fine-tuning-of-llama-for-ccks2023-instruction-based-kg-construction-english--chinese)
    - [Case 2: Using ZhiXi for CCKS2023 Instruction-based KG Construction English | Chinese](#case-2-using-zhixi-for-ccks2023-instruction-based-kg-construction-english--chinese)
  - [ChatGLM](#chatglm)
    - [Case 1: LoRA Fine-tuning of ChatGLM for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-lora-fine-tuning-of-chatglm-for-ccks2023-instruction-based-kg-construction-english--chinese)
    - [Case 2: P-Tuning of ChatGLM for CCKS2023 Instruction-based KG Construction English | Chinese](#case-2-p-tuning-of-chatglm-for-ccks2023-instruction-based-kg-construction-english--chinese)
  - [MOSS](#moss)
    - [Case 1: OpenDelta Fine-tuning of Moss for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-opendelta-fine-tuning-of-moss-for-ccks2023-instruction-based-kg-construction-english--chinese)
  - [Baichuan](#baichuan)
    - [Case 1: OpenDelta Fine-tuning of Baichuan for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-opendelta-fine-tuning-of-baichuan-for-ccks2023-instruction-based-kg-construction-english--chinese)
  - [CPM-Bee](#cpm-bee)
    - [Case 1: OpenDelta Fine-tuning of CPM-Bee for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-opendelta-fine-tuning-of-cpm-bee-for-ccks2023-instruction-based-kg-construction-english--chinese)
  - [GPT-series](#gpt-series)
    - [Case 1: Information Extraction with LLMs English | Chinese](#case-1-information-extraction-with-llms-english--chinese)
    - [Case 2: Data Augmentation with LLMs English | Chinese](#case-2-data-augmentation-with-llms-english--chinese)
    - [Case 3: CCKS2023 Instruction-based KG Construction with LLMs English | Chinese](#case-3-ccks2023-instruction-based-kg-construction-with-llms-english--chinese)
    - [Case 4: Unleash the Power of Large Language Models for Few-shot Relation Extraction English | Chinese](#case-4-unleash-the-power-of-large-language-models-for-few-shot-relation-extraction-english--chinese)
    - [Case 5: CodeKGC-Code Language Models for KG Construction English | Chinese](#case-5-codekgc-code-language-models-for-kg-construction-english--chinese)
- [Methods](#methods)
  - [Method 1: In-Context Learning (ICL)](#method-1-in-context-learning-icl)
  - [Method 2: LoRA](#method-2-lora)
  - [Method 3: P-Tuning](#method-3-p-tuning)
- [Citation](#citation)


# Requirements

In the era of large models, DeepKE-LLM utilizes a completely new environment dependency.

```
conda create -n deepke-llm python=3.9
conda activate deepke-llm

cd example/llm
pip install -r requirements.txt
```

Please note that the `requirements.txt` file is located in the `example/llm` folder.



# Data
| Name                   | Download                                                     | Quantity | Description                                                  |
| ---------------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| InstructIE-train          | [Google drive](https://drive.google.com/file/d/1VX5buWC9qVeVuudh_mhc_nC7IPPpGchQ/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/1xXVrjkinw4cyKKFBR8BwQw?pwd=x4s7)  | 30w+  | InstructIE train set, which is constructed by weak supervision and may contain some noisy data |
| InstructIE-valid       | [Google drive](https://drive.google.com/file/d/1EMvqYnnniKCGEYMLoENE1VD6DrcQ1Hhj/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/11u_f_JT30W6B5xmUPC3enw?pwd=71ie)     | 2000+ | InstructIE validation set                                                                                     |
| InstructIE-test       | [Google drive](https://drive.google.com/file/d/1WdG6_ouS-dBjWUXLuROx03hP-1_QY5n4/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)  <br/> [Baidu Netdisk](https://pan.baidu.com/s/1JiRiOoyBVOold58zY482TA?pwd=cyr9)     | 2000+ | InstructIE test set                                                                                    |
| train.json, valid.json | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing) | 5,000    | Preliminary training set and test set for the task "Instruction-Driven Adaptive Knowledge Graph Construction" in [CCKS2023 Open Knowledge Graph Challenge](https://tianchi.aliyun.com/competition/entrance/532080/introduction), randomly selected from instruct_train.json |


`InstrumentIE-train` contains two files: `InstrumentIE-zh.json` and `InstrumentIE-en.json`, each of which contains the following fields: `'id'` (unique identifier), `'cate'` (text category), `'entity'` and `'relation'` (triples) fields. The extracted instructions and output can be freely constructed through `'entity'` and `'relation'`.

`InstrumentIE-valid` and `InstrumentIE-test` are validation sets and test sets, respectively, including bilingual `zh` and `en`.

`train.json`: Same fields as `KnowLM-IE.json`, `'instruction'` and `'output'` have only one format, and extraction instructions and outputs can also be freely constructed through `'relation'`.

`valid.json`: Same fields as `train.json`, but with more accurate annotations achieved through crowdsour

Here is an explanation of each field:

|    Field    |                         Description                             |
| :---------: |   :----------------------------------------------------------:  |
|     id      |                   Unique identifier                             |
|    cate     |     text topic of input (12 topics in total)                    |
|    input    | Model input text (need to extract all triples involved within)  |
| instruction |   Instruction for the model to perform the extraction task      |
|    output   |                  Expected model output                          |
|    entity   |            entities(entity, entity_type)                        |
|   relation  |   Relation triples(head, relation, tail) involved in the input  |



# Models

## LLaMA-series

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/llama.jpg" alt="LLaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>

LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We also provide a bilingual LLM for knowledge extraction named `ZhiXi (智析)` (which means intelligent analysis of data for knowledge extraction) based on [KnowLM](https://github.com/zjunlp/KnowLM).

ZhiXi follows a two-step approach: (1) It performs further full pre-training on `LLaMA (13B)` using Chinese/English corpora to enhance the model's Chinese comprehension and knowledge while preserving its English and code capabilities as much as possible. (2) It fine-tunes the model using an instruction dataset to improve the language model's understanding of human instructions. For detailed information about the model, please refer to [KnowLM](https://github.com/zjunlp/KnowLM).


### Case 1: LoRA Fine-tuning of LLaMA for CCKS2023 Instruction-based KG Construction [English](./InstructKGC/README.md/#42lora-fine-tuning-with-llama) | [Chinese](./InstructKGC/README_CN.md/#42lora微调llama)

### Case 2: Using ZhiXi for CCKS2023 Instruction-based KG Construction [English](./InstructKGC/README.md/#44lora-fine-tuning-with-zhixi-智析) | [Chinese](./InstructKGC/README_CN.md/#44lora微调智析)




## ChatGLM
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatglm_logo.png" alt="ChatGLM" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### Case 1: LoRA Fine-tuning of ChatGLM for CCKS2023 Instruction-based KG Construction [English](./InstructKGC//README.md/#46lora-fine-tuning-with-chatglm) | [Chinese](./InstructKGC//README_CN.md/#46lora微调chatglm) 
### Case 2: P-Tuning of ChatGLM for CCKS2023 Instruction-based KG Construction [English](./InstructKGC/README.md/#51p-tuning-fine-tuning-with-chatglm) | [Chinese](./InstructKGC/README_CN.md/#51p-tuning微调chatglm)




## MOSS
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/moss_logo.png" alt="ChatGLM" style="width: 25%; min-width: 25px; display: block; margin: auto;"></a>


### Case 1: OpenDelta Fine-tuning of Moss for CCKS2023 Instruction-based KG Construction [English](./InstructKGC//README.md/#47lora-fine-tuning-with-moss) | [Chinese](./InstructKGC//README_CN.md/#47lora微调moss) 





## Baichuan
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/baichuan_logo.png" alt="Baichuan" style="width: 25%; min-width: 25px; display: block; margin: auto;"></a>


### Case 1: OpenDelta Fine-tuning of Baichuan for CCKS2023 Instruction-based KG Construction [English](./InstructKGC//README.md/#48lora-fine-tuning-with-baichuan) | [Chinese](./InstructKGC//README_CN.md/#48lora微调baichuan) 





## CPM-Bee
### Case 1: OpenDelta Fine-tuning of CPM-Bee for CCKS2023 Instruction-based KG Construction [English](./CPM-Bee/README.md) | [Chinese](./CPM-Bee/README_CN.md) 



## GPT-series

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatgpt_logo.png" alt="GPT" style="width: 18%; min-width: 18px; display: block; margin: auto;"></a>
</p>


### Case 1: Information Extraction with LLMs [English](./LLMICL/README.md/#ie-with-large-language-models) | [Chinese](./LLMICL/README_CN.md/#使用大语言模型进行信息抽取)


### Case 2: Data Augmentation with LLMs [English](./LLMICL/README.md/#data-augmentation-with-large-language-models) | [Chinese](./LLMICL/README_CN.md/#使用大语言模型进行数据增强)



### Case 3: CCKS2023 Instruction-based KG Construction with LLMs [English](./LLMICL/README.md/#ccks2023-instruction-based-knowledge-graph-construction-with-large-language-models) | [Chinese](./LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建)

### Case 4: Unleash the Power of Large Language Models for Few-shot Relation Extraction [English](./UnleashLLMRE/README.md) | [Chinese](./UnleashLLMRE/README_CN.md)

### Case 5: CodeKGC-Code Language Models for KG Construction [English](./CodeKGC/README.md) | [Chinese](./CodeKGC/README_CN.md)

To better address Relational Triple Extraction (rte) task in Knowledge Graph Construction, we have designed code-style prompts to model the structure of  Relational Triple, and used Code-LLMs to generate more accurate predictions. The key step of code-style prompt construction is to transform (text, output triples) pairs into semantically equivalent program language written in Python.

---

# Methods

## Method 1: In-Context Learning (ICL)
[In-Context Learning](http://arxiv.org/abs/2301.00234) is an approach to guide large language models to improve their performance on specific tasks. It involves iterative fine-tuning and training of the model in a specific context to better understand and address the requirements of a particular domain. Through In-Context Learning, we can enable large language models to perform tasks such as information extraction, data augmentation, and instruction-driven knowledge graph construction.

## Method 2: LoRA
LoRA (Low-Rank Adaptation of Large Language Models) reduces the number of trainable parameters by learning low-rank decomposition matrices while freezing the original weights. This significantly reduces the storage requirements of large language models for specific tasks and enables efficient task switching during deployment without introducing inference latency. For more details, please refer to the original paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

## Method 3: P-Tuning
The PT (P-Tuning) method, as referred to in the official code of ChatGLM, is a soft-prompt method specifically designed for large models. 
[P-Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.10385) introduces new parameters only to the embeddings of large models. 
[P-Tuning-V2](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2110.07602) adds new parameters to both the embeddings and the preceding layers of large models.

# Citation

If you use this project, please cite the following papers:

```bibtex
@misc{knowlm,
  author = {Ningyu Zhang and Jintian Zhang and Xiaohan Wang and Honghao Gui and Kangwei Liu and Yinuo Jiang and Xiang Chen and Shengyu Mao and Shuofei Qiao and Yuqi Zhu and Zhen Bi and Jing Chen and Xiaozhuan Liang and Yixin Ou and Runnan Fang and Zekun Xi and Xin Xu and Lei Li and Peng Wang and Mengru Wang and Yunzhi Yao and Bozhong Tian and Yin Fang and Guozhou Zheng and Huajun Chen},
  title = {KnowLM Technical Report},
  year = {2023},
 url = {http://knowlm.zjukg.cn/},
}
```
