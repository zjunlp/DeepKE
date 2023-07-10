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
- [Models](#models)
  - [LLaMA-series](#llama-series)
    - [Case 1: LoRA Fine-tuning of LLaMA for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-lora-fine-tuning-of-llama-for-ccks2023-instruction-based-kg-construction-english--chinese)
    - [Case 2: Using ZhiXi for CCKS2023 Instruction-based KG Construction English | Chinese](#case-2-using-zhixi-for-ccks2023-instruction-based-kg-construction-english--chinese)
  - [ChatGLM](#chatglm)
    - [Case 1: LoRA Fine-tuning of ChatGLM for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-lora-fine-tuning-of-chatglm-for-ccks2023-instruction-based-kg-construction-english--chinese)
    - [Case 2: P-Tuning of ChatGLM for CCKS2023 Instruction-based KG Construction English | Chinese](#case-2-p-tuning-of-chatglm-for-ccks2023-instruction-based-kg-construction-english--chinese)
  - [MOSS](#moss)
    - [Case 1: OpenDelta Fine-tuning of Moss for CCKS2023 Instruction-based KG Construction English | Chinese](#case-1-opendelta-fine-tuning-of-moss-for-ccks2023-instruction-based-kg-construction-english--chinese)
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



# Models

## LLaMA-series

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/llama.jpg" alt="LLaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>

LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We also provide a bilingual LLM for knowledge extraction named `ZhiXi (智析)` (which means intelligent analysis of data for knowledge extraction) based on [KnowLM](https://github.com/zjunlp/KnowLM).

ZhiXi follows a two-step approach: (1) It performs further full pre-training on `LLaMA (13B)` using Chinese/English corpora to enhance the model's Chinese comprehension and knowledge while preserving its English and code capabilities as much as possible. (2) It fine-tunes the model using an instruction dataset to improve the language model's understanding of human instructions. For detailed information about the model, please refer to [KnowLM](https://github.com/zjunlp/KnowLM).


### Case 1: LoRA Fine-tuning of LLaMA for CCKS2023 Instruction-based KG Construction [English](./InstructKGC/README.md/#lora-fine-tuning-with-llama) | [Chinese](./InstructKGC/README_CN.md/#lora微调llama)

### Case 2: Using ZhiXi for CCKS2023 Instruction-based KG Construction [English](./InstructKGC/README.md/#lora-fine-tuning-with-zhixi-智析) | [Chinese](./InstructKGC/README_CN.md/#lora微调智析)




## ChatGLM
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatglm_logo.png" alt="ChatGLM" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### Case 1: LoRA Fine-tuning of ChatGLM for CCKS2023 Instruction-based KG Construction [English](./InstructKGC//README.md/#lora-fine-tuning-with-chatglm) | [Chinese](./InstructKGC//README_CN.md/#lora微调chatglm) 
### Case 2: P-Tuning of ChatGLM for CCKS2023 Instruction-based KG Construction [English](./InstructKGC/README.md/#p-tuning-fine-tuning-with-chatglm) | [Chinese](./InstructKGC/README_CN.md/#p-tuning微调chatglm)




## MOSS
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/moss_logo.png" alt="ChatGLM" style="width: 25%; min-width: 25px; display: block; margin: auto;"></a>


### Case 1: OpenDelta Fine-tuning of Moss for CCKS2023 Instruction-based KG Construction [English](./InstructKGC//README.md/#lora-fine-tuning-with-moss) | [Chinese](./InstructKGC//README_CN.md/#lora微调moss) 




## CPM-Bee
### Case 1: OpenDelta Fine-tuning of CPM-Bee for CCKS2023 Instruction-based KG Construction [English](./InstructKGC//cpm-bee//README.md) | [Chinese](./InstructKGC//cpm-bee//README_CN.md) 



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
@article{deepke-llm,
  author = {Ningyu Zhang, Jintian Zhang, Xiaohan Wang, Honghao Gui, Yinuo Jiang, Xiang Chen, Shengyu Mao, Shuofei Qiao, Zhen Bi, Jing Chen, Xiaozhuan Liang, Yixin Ou, Ruinan Fang, Zekun Xi, Xin Xu, Liankuan Tao, Lei Li, Peng Wang, Zhoubo Li, Guozhou Zheng, Huajun Chen},
  title = {DeepKE-LLM: A Large Language Model Based Knowledge Extraction Toolkit},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/}},
}
```
