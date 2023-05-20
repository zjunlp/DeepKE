<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README.md">English</a> | 简体中文</a> </b>
</p>

## 目录

- [目录](#目录)
- [使用In-Context Learning指导大语言模型](#使用in-context-learning指导大语言模型)
  - [使用大语言模型进行信息抽取 英文 | 中文](#使用大语言模型进行信息抽取-英文--中文)
  - [使用大语言模型进行数据增强 英文 | 中文](#使用大语言模型进行数据增强-英文--中文)
  - [使用大语言模型完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#使用大语言模型完成ccks2023指令驱动的知识图谱构建-英文--中文)
- [CodeKGC-基于代码语言模型的知识图谱构建 英文 | 中文](#codekgc-基于代码语言模型的知识图谱构建-英文--中文)


## 使用In-Context Learning指导大语言模型
[In-Context Learning](http://arxiv.org/abs/2301.00234) 是一种指导大语言模型的方法，可以提升其在特定任务上的表现。它通过在特定上下文中进行迭代学习，对模型进行微调和训练，以使其更好地理解和应对特定领域的需求。通过 `In-Context Learning`，我们可以让大语言模型具备信息抽取、数据增强以及指令驱动的知识图谱构建等功能。


### 使用大语言模型进行信息抽取 [英文](./LLMICL/README.md/#ie-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行信息抽取)
了解如何使用大语言模型进行 `信息抽取`。通过指定特定的输入和输出格式，模型可以提取文本中的关键信息，并将其转化为结构化的数据。

### 使用大语言模型进行数据增强 [英文](./LLMICL/README.md/#data-augmentation-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型进行数据增强)
探索如何利用大语言模型进行 `数据增强`。为了弥补少样本场景下关系抽取有标签数据的缺失, 我们设计带有数据样式描述的提示，用于指导大型语言模型根据已有的少样本数据自动地生成更多的有标签数据。

### 使用大语言模型完成CCKS2023指令驱动的知识图谱构建 [英文](./LLMICL/README.md/#ccks2023-instruction-based-knowledge-graph-construction-with-large-language-models) | [中文](./LLMICL/README_CN.md/#使用大语言模型完成ccks2023指令驱动的知识图谱构建)
了解如何利用大语言模型完成 `CCKS2023指令驱动的知识图谱构建任务`。通过将指令输入给模型，它可以学习并生成符合指定要求的知识图谱，从而提供更全面和有价值的知识表示。


## CodeKGC-基于代码语言模型的知识图谱构建 [英文](./CodeKGC/README.md) | [中文](./CodeKGC/README_CN.md)

为了更好地处理知识图谱构建中的关系三元组抽取（RTE）任务，我们设计了代码形式的提示建模关系三元组的结构，并使用代码语言模型（Code-LLM）生成更准确的预测。代码形式提示构建的关键步骤是将（文本，输出三元组）对转换成Python中的语义等价的程序语言。

<div align=center>
<img src="./CodeKGC/codekgc_figure.png" width="85%" height="75%" />
</div>
