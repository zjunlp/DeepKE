<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/llm/example/llm/README.md">English</a> | 简体中文</a> </b>
</p>

# 目录

- [目录](#目录)
- [方法](#方法)
  - [方法一：In-Context Learning(ICL)](#方法一in-context-learningicl)
  - [方法二：LoRA](#方法二lora)
  - [方法三：P-Tuning](#方法三p-tuning)
- [模型](#模型)
  - [CaMA](#cama)
    - [案例一：LoRA微调CaMA完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调cama完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [LLaMA](#llama)
    - [案例一：CodeKGC-基于代码语言模型的知识图谱构建 英文 | 中文](#案例一codekgc-基于代码语言模型的知识图谱构建-英文--中文)
    - [案例二：LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例二lora微调llama完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [ChatGLM](#chatglm)
    - [案例一：LoRA微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调chatglm完成ccks2023指令驱动的知识图谱构建-英文--中文)
    - [案例一：P-Tuning微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一p-tuning微调chatglm完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [GPT系列](#gpt系列)
    - [案例一：ICL使用大语言模型进行信息抽取 英文 | 中文](#案例一icl使用大语言模型进行信息抽取-英文--中文)
    - [案例二：ICL使用大语言模型进行数据增强 英文 | 中文](#案例二icl使用大语言模型进行数据增强-英文--中文)
    - [案例三：ICL使用大语言模型完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例三icl使用大语言模型完成ccks2023指令驱动的知识图谱构建-英文--中文)
    - [案例四：ICL释放大型语言模型进行少样本关系抽取中的能力 英文 | 中文](#案例四icl释放大型语言模型进行少样本关系抽取中的能力-英文--中文)

---

# 方法

## 方法一：In-Context Learning(ICL)
[In-Context Learning](http://arxiv.org/abs/2301.00234) 是一种指导大语言模型的方法，可以提升其在特定任务上的表现。它通过在特定上下文中进行迭代学习，对模型进行微调和训练，以使其更好地理解和应对特定领域的需求。通过 `In-Context Learning`，我们可以让大语言模型具备信息抽取、数据增强以及指令驱动的知识图谱构建等功能。

## 方法二：LoRA
LoRA通过学习秩分解矩阵对来减少可训练参数的数量，同时冻结原始权重。这大大降低了适用于特定任务的大型语言模型的存储需求，并在部署期间实现了高效的任务切换，而无需引入推理延迟。详细可查看原论文[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

## 方法三：P-Tuning
PT方法，即P-Tuning方法，参考[ChatGLM官方代码](https://link.zhihu.com/?target=https%3A//github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md) ，是一种针对于大模型的soft-prompt方法。
[P-Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.10385)，仅对大模型的Embedding加入新的参数。
[P-Tuning-V2](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2110.07602)，将大模型的Embedding和每一层前都加上新的参数。


# 模型

## CaMA

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/cama_logo.jpeg" alt="ZJU-CaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### 案例一：LoRA微调CaMA完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md) | [中文](./InstructKGC/README_CN.md)


---

## LLaMA

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/llama_logo.jpeg" alt="LLaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>



### 案例一：CodeKGC-基于代码语言模型的知识图谱构建 [英文](./CodeKGC/README.md) | [中文](./CodeKGC/README_CN.md)

为了更好地处理知识图谱构建中的关系三元组抽取（RTE）任务，我们设计了代码形式的提示建模关系三元组的结构，并使用代码语言模型（Code-LLM）生成更准确的预测。代码形式提示构建的关键步骤是将（文本，输出三元组）对转换成Python中的语义等价的程序语言。

### 案例二：LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md) | [中文](./InstructKGC/README_CN.md)


---

## ChatGLM
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatglm_logo.png" alt="ChatGLM" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### 案例一：LoRA微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md) | [中文](./InstructKGC/README_CN.md)
### 案例一：P-Tuning微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md) | [中文](./InstructKGC/README_CN.md)

---

## GPT系列

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatgpt_logo.png" alt="GPT" style="width: 18%; min-width: 18px; display: block; margin: auto;"></a>
</p>



### 案例一：ICL使用大语言模型进行信息抽取 [英文](./LLMICL/README.md/#ie-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行信息抽取)


### 案例二：ICL使用大语言模型进行数据增强 [英文](./LLMICL/README.md/#data-augmentation-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行数据增强)


### 案例三：ICL使用大语言模型完成CCKS2023指令驱动的知识图谱构建 [英文](./LLMICL/README.md/#ccks2023-instruction-based-knowledge-graph-construction-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建)

### 案例四：ICL释放大型语言模型进行少样本关系抽取中的能力 [英文](./UnleashLLMRE/README.md) | [中文](./UnleashLLMRE/README_CN.md)


