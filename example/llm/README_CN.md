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
- [模型](#模型)
  - [LLaMA系列](#llama系列)
    - [案例一：LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调llama完成ccks2023指令驱动的知识图谱构建-英文--中文)
    - [案例二：使用智析完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例二使用智析完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [ChatGLM](#chatglm)
    - [案例一：LoRA微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调chatglm完成ccks2023指令驱动的知识图谱构建-英文--中文)
    - [案例二：P-Tuning微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例二p-tuning微调chatglm完成ccks2023指令驱动的知识图谱构建-英文--中文)
  - [MOSS](#moss)
    - [案例一：Lora微调Moss完成CCKS2023指令驱动的知识图谱构建 英文 | 中文](#案例一lora微调moss完成ccks2023指令驱动的知识图谱构建-英文--中文)
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

# 模型



## LLaMA系列
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/llama.jpg" alt="LLaMA" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>

LLaMA，它是一组从7B到65B参数的基础语言模型。我们还提供了基于[KnowLM](https://github.com/zjunlp/KnowLM)框架的抽取大模型`智析`的支持。其首先（1）使用中英双语语料首先对LLaMA（13B）进行进一步全量预训练，在尽可能保留原来的英文和代码能力的前提下，进一步提高模型对于中文理解能力和知识储备；接着（2）使用指令数据集对第一步的模型微调，来提高语言模型对于人类指令的理解。模型详细信息请参考[KnowLM](https://github.com/zjunlp/KnowLM).

### 案例一：LoRA微调LLaMA完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md/#lora-fine-tuning-with-llama) | [中文](./InstructKGC/README_CN.md/#lora微调llama)

### 案例二：使用智析完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md/#lora-fine-tuning-with-zhixi-智析) | [中文](./InstructKGC/README_CN.md/#lora微调智析)



## ChatGLM
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/chatglm_logo.png" alt="ChatGLM" style="width: 20%; min-width: 20px; display: block; margin: auto;"></a>
</p>

### 案例一：LoRA微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC//README.md/#lora-fine-tuning-with-chatglm) | [中文](./InstructKGC//README_CN.md/#lora微调chatglm) 
### 案例二：P-Tuning微调ChatGLM完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC/README.md/#lora-fine-tuning-with-chatglm) | [中文](./InstructKGC/README_CN.md/#p-tuning微调chatglm)




## MOSS
<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/moss_logo.png" alt="ChatGLM" style="width: 25%; min-width: 25px; display: block; margin: auto;"></a>

### 案例一：Lora微调Moss完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC//README.md/#lora-fine-tuning-with-moss) | [中文](./InstructKGC//README_CN.md/#lora微调Moss) 




## CPM-Bee
### 案例一：OpenDelta微调CPM-Bee完成CCKS2023指令驱动的知识图谱构建 [英文](./InstructKGC//cpm-bee//README.md) | [中文](./InstructKGC//cpm-bee//README_CN.md) 




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
@article{deepke-llm,
  author = {Ningyu Zhang, Jintian Zhang, Xiaohan Wang, Honghao Gui, Yinuo Jiang, Xiang Chen, Shengyu Mao, Shuofei Qiao, Zhen Bi, Jing Chen, Xiaozhuan Liang, Yixin Ou, Ruinan Fang, Zekun Xi, Xin Xu, Liankuan Tao, Lei Li, Peng Wang, Zhoubo Li, Guozhou Zheng, Huajun Chen},
