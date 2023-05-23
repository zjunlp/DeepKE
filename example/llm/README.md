<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/llm/example/llm/README_CN.md">简体中文</a> </b>
</p>

- [CaMA](#cama)
  - [Training](#training)
  - [Usage](#usage)
- [LLaMA](#llama)
- [ChatGLM](#chatglm)
- [GPT series](#gpt-series)
  - [Instruct Large Language Models using In-Context Learning](#instruct-large-language-models-using-in-context-learning)
    - [Information Extraction English | Chinese](#information-extraction-english--chinese)
    - [Data Augmentation English | Chinese](#data-augmentation-english--chinese)
    - [CCKS2023 Instruction-based Knowledge Graph Construction English | Chinese](#ccks2023-instruction-based-knowledge-graph-construction-english--chinese)
- [CodeKGC-Code Language Models for Knowledge Graph Construction English | Chinese](#codekgc-code-language-models-for-knowledge-graph-construction-english--chinese)

---

# CaMA

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/cama_logo.jpeg" alt="ZJU-CaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

## Training

## Usage
 
---

# LLaMA

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/llama_logo.jpeg" alt="LLaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

--- 

# ChatGLM
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatglm_logo.png" alt="ChatGLM" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

---

# GPT series

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatgpt_logo.png" alt="GPT" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

## Instruct Large Language Models using In-Context Learning

[In-Context Learning](http://arxiv.org/abs/2301.00234) is a method for instructing large language models to improve their performance on specific tasks. It involves iterative learning within specific contexts to fine-tune and train the model, enabling it to better understand and respond to the demands of a particular domain. Through In-Context Learning, we can empower large language models with capabilities such as information extraction, data augmentation, and instruction-driven knowledge graph construction.


### Information Extraction [English](./LLMICL/README.md/#ie-with-large-language-models) | [Chinese](./LLMICL/README_CN.md/#使用大语言模型进行信息抽取)

Learn how to perform information extraction using large language models. By specifying specific input and output formats, the model can extract key information from text and convert it into structured data.


### Data Augmentation [English](./LLMICL/README.md/#data-augmentation-with-large-language-models) | [Chinese](./LLMICL/README_CN.md/#使用大语言模型进行数据增强)

Explore how to leverage large language models for data augmentation. To address the lack of labeled data in low-resource scenarios for relation extraction, we design prompts with data style descriptions to guide the large language model in automatically generating more labeled data based on the existing few-shot data.


### CCKS2023 Instruction-based Knowledge Graph Construction [English](./LLMICL/README.md/#ccks2023-instruction-based-knowledge-graph-construction-with-large-language-models) | [Chinese](./LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建)

Learn how to utilize large language models to accomplish the task of [CCKS2023 Instruction-Based Knowledge Graph Construction](https://tianchi.aliyun.com/competition/entrance/532080/introduction) By inputting instructions to the model, it can learn and generate knowledge graphs that meet specified requirements, thereby providing more comprehensive and valuable knowledge representations.


# CodeKGC-Code Language Models for Knowledge Graph Construction [English](./CodeKGC/README.md) | [Chinese](./CodeKGC/README_CN.md)

To better address Relational Triple Extraction (rte) task in Knowledge Graph Construction, we have designed code-style prompts to model the structure of  Relational Triple, and used Code-LLMs to generate more accurate predictions. The key step of code-style prompt construction is to transform (text, output triples) pairs into semantically equivalent program language written in Python.

<div align=center>
<img src="./CodeKGC/codekgc_figure.png" width="85%" height="75%" />
</div>



