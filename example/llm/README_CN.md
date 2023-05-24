<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/llm/example/llm/README.md">English</a> | 简体中文</a> </b>
</p>

# 目录

- [目录](#目录)
  - [CaMA](#cama)
    - [1.训练](#1训练)
    - [2.用法](#2用法)
  - [LLaMA](#llama)
    - [方法一：LoRA微调](#方法一lora微调)
      - [案例一：CodeKGC-基于代码语言模型的知识图谱构建 英文 | 中文](#案例一codekgc-基于代码语言模型的知识图谱构建-英文--中文)
      - [案例二：使用LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例二使用lora微调llama完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [ChatGLM](#chatglm)
  - [GPT系列](#gpt系列)
    - [方法一：使用In-Context Learning指导大语言模型](#方法一使用in-context-learning指导大语言模型)
      - [案例一：使用大语言模型进行信息抽取 英文 | 中文](#案例一使用大语言模型进行信息抽取-英文--中文)
      - [案例二：使用大语言模型进行数据增强 英文 | 中文](#案例二使用大语言模型进行数据增强-英文--中文)
      - [案例三：使用大语言模型完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例三使用大语言模型完成ccks2023指令驱动的知识图谱构建-英文--中文)
      - [案例四：释放大型语言模型进行少样本关系抽取中的能力 英文 | 中文](#案例四释放大型语言模型进行少样本关系抽取中的能力-英文--中文)

---

## CaMA

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/cama_logo.jpeg" alt="ZJU-CaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### 1.训练


### 2.用法 

---

## LLaMA

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/llama_logo.jpeg" alt="LLaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### 方法一：LoRA微调

LoRA通过学习秩分解矩阵对来减少可训练参数的数量，同时冻结原始权重。这大大降低了适用于特定任务的大型语言模型的存储需求，并在部署期间实现了高效的任务切换，而无需引入推理延迟。详细可查看原论文[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

#### 案例一：CodeKGC-基于代码语言模型的知识图谱构建 [英文](./CodeKGC/README.md) | [中文](./CodeKGC/README_CN.md)

为了更好地处理知识图谱构建中的关系三元组抽取（RTE）任务，我们设计了代码形式的提示建模关系三元组的结构，并使用代码语言模型（Code-LLM）生成更准确的预测。代码形式提示构建的关键步骤是将（文本，输出三元组）对转换成Python中的语义等价的程序语言。

#### 案例二：使用LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md) | [中文](./InstructKGC/README_CN.md)


---

## ChatGLM
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatglm_logo.png" alt="ChatGLM" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

---

## GPT系列

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatgpt_logo.png" alt="GPT" style="width: 18%; min-width: 18px; display: block; margin: auto;"></a>
</p>

### 方法一：使用In-Context Learning指导大语言模型
[In-Context Learning](http://arxiv.org/abs/2301.00234) 是一种指导大语言模型的方法，可以提升其在特定任务上的表现。它通过在特定上下文中进行迭代学习，对模型进行微调和训练，以使其更好地理解和应对特定领域的需求。通过 `In-Context Learning`，我们可以让大语言模型具备信息抽取、数据增强以及指令驱动的知识图谱构建等功能。


#### 案例一：使用大语言模型进行信息抽取 [英文](./LLMICL/README.md/#ie-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行信息抽取)


#### 案例二：使用大语言模型进行数据增强 [英文](./LLMICL/README.md/#data-augmentation-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行数据增强)


#### 案例三：使用大语言模型完成CCKS2023指令驱动的知识图谱构建 [英文](./LLMICL/README.md/#ccks2023-instruction-based-knowledge-graph-construction-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建)

#### 案例四：释放大型语言模型进行少样本关系抽取中的能力 [英文](./UnleashLLMRE/README.md) | [中文](./UnleashLLMRE/README_CN.md)


