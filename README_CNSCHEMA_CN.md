
<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="pics/logo.png" width="400"/></a>
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


<h1 align="center">
    <p>基于深度学习的开源中文知识图谱抽取框架</p>
</h1>


DeepKE 是一个开源的知识图谱抽取与构建工具，支持<b>低资源、长篇章</b>的知识抽取工具，可以基于<b>PyTorch</b>实现<b>命名实体识别</b>、<b>关系抽取</b>和<b>属性抽取</b>功能。同时为初学者提供了详尽的[文档](https://zjunlp.github.io/DeepKE/)，[Google Colab教程](https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing)和[在线演示](http://deepke.zjukg.cn/CN/index.html)。

----


## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍DeepKE-cnSchema基本原理 |
| [中文模型下载](#中文模型下载) | 提供了DeepKE-cnSchema的下载地址 |
| [快速加载](#快速加载) | 介绍了如何使用DeepKE-cnSchema进行实体识别、关系抽取 |
| [中文模型效果](#中文基线系统效果) | 列举了中文模型效果 |
| [使用建议](#使用建议) | 提供了若干使用中文预训练模型的建议 |
| [自定义模型](#高级用户) | 提供了使用自定义数据训练模型的说明 |
| [FAQ](#FAQ) | 常见问题答疑 |
| [引用](#引用) | 本目录的技术报告 |


## 简介

DeepKE 是一个开源的知识图谱抽取与构建工具，支持低资源、长篇章、多模态的知识抽取工具，可以基于PyTorch实现命名实体识别、关系抽取和属性抽取功能。同时为初学者提供了详尽的文档，Google Colab教程和在线演示。

为了促进中文领域的知识图谱构建，DeepKE提供了预训练好的支持cnSchema的开箱即用的抽取特别版DeepKE-cnSchema，支持开箱即用的中文实体识别和关系预测功能。该版本支持抽取包含了50种不同的关系类型以及28种不同的实体类型，其中实体类型包含了通用的人物、地点、城市、机构等类型，关系类型包括了常见的祖籍、出生地、国籍、朝代等类型。

## 中文模型下载
本目录中主要包含base模型，故我们不在模型简称中标注`base`字样。

* **`base模型`**：12-layer, 768-hidden, 12-heads, 110M parameters  


| 模型简称 | 功能 | Google下载 | 百度网盘下载 |
| :------- | :--------- | :---------: | :---------: |
| **`DeepKE(NER), RoBERTa-wwm-ext, Chinese`** | **EXT数据<sup>[1]</sup>** | **[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[Pytorch（密码u022）](https://pan.baidu.com/s/1hb9XEbK4x5fIyco4DgZZfg)** |


### 使用说明

中国大陆境内建议使用百度网盘下载点，境外用户建议使用谷歌下载点，base模型文件大小约**400M**。 
以Pytorch版`BERT-wwm, Chinese`为例，下载完毕后对zip文件进行解压得到：

```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # 模型权重
    |- bert_model.meta      # 模型meta信息
    |- bert_model.index     # 模型index信息
    |- bert_config.json     # 模型参数
    |- vocab.txt            # 词表
```
其中`bert_config.json`和`vocab.txt`与谷歌原版`BERT-base, Chinese`完全一致。
PyTorch版本则包含`pytorch_model.bin`, `bert_config.json`, `vocab.txt`文件。


## 快速加载
### 使用Huggingface-Transformers

依托于[🤗transformers库](https://github.com/huggingface/transformers)，可轻松调用以上模型。
```
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```
**注意：本目录中的所有模型均使用BertTokenizer以及BertModel加载，请勿使用RobertaTokenizer/RobertaModel！**

其中`MODEL_NAME`对应列表如下：

| 模型名 | MODEL_NAME |
| - | - |
| RoBERTa-wwm-ext-large | hfl/chinese-roberta-wwm-ext-large |
| RoBERTa-wwm-ext | hfl/chinese-roberta-wwm-ext |
| BERT-wwm-ext | hfl/chinese-bert-wwm-ext |
| BERT-wwm | hfl/chinese-bert-wwm |
| RBT3 | hfl/rbt3 |
| RBTL3 | hfl/rbtl3 |


## 中文基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试，包括`句子级`和`篇章级`任务。
对于`BERT-wwm-ext`、`RoBERTa-wwm-ext`、`RoBERTa-wwm-ext-large`，我们**没有进一步调整最佳学习率**，而是直接使用了`BERT-wwm`的最佳学习率。

 

### 简体中文阅读理解：CMRC 2018

[cnSchema](https://github.com/OpenKG-ORG/cnSchema)

[**CMRC 2018数据集**](https://github.com/ymcui/cmrc2018)是哈工大讯飞联合实验室发布的中文机器阅读理解数据。
根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。
评测指标为：EM / F1

| 模型 | 开发集 | 测试集 | 挑战集 |
| :------- | :---------: | :---------: | :---------: |
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) |
| ERNIE | 65.4 (64.3) / 84.7 (84.2) | 69.4 (68.2) / 86.6 (86.1) | 19.6 (17.0) / 44.3 (42.8) |
| **BERT-wwm** | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) |
| **BERT-wwm-ext** | 67.1 (65.6) / 85.7 (85.0) | 71.4 (70.0) / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| **RoBERTa-wwm-ext** | 67.4 (66.5) / 87.2 (86.5) | 72.6 (71.4) / 89.4 (88.8) | 26.2 (24.6) / 51.0 (49.1) |
| **RoBERTa-wwm-ext-large** | **68.5 (67.6) / 88.4 (87.9)** | **74.2 (72.4) / 90.6 (90.0)** | **31.5 (30.1) / 60.1 (57.5)** |


### 繁体中文阅读理解：DRCD
[**DRCD数据集**](https://github.com/DRCKnowledgeTeam/DRCD)由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。
**由于ERNIE中去除了繁体中文字符，故不建议在繁体中文数据上使用ERNIE（或转换成简体中文后再处理）。**
评测指标为：EM / F1

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) |
| ERNIE | 73.2 (73.0) / 83.9 (83.8) | 71.9 (71.4) / 82.5 (82.3) |
| **BERT-wwm** | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) |
| **BERT-wwm-ext** | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) |
| **RoBERTa-wwm-ext** | 86.6 (85.9) / 92.5 (92.2) | 85.6 (85.2) / 92.0 (91.7) |
| **RoBERTa-wwm-ext-large** | **89.6 (89.1) / 94.8 (94.4)** | **89.6 (88.9) / 94.5 (94.1)** |




## 使用建议
* 初始学习率是非常重要的一个参数（不论是`BERT`还是其他模型），需要根据目标任务进行调整。
* `ERNIE`的最佳学习率和`BERT`/`BERT-wwm`相差较大，所以使用`ERNIE`时请务必调整学习率（基于以上实验结果，`ERNIE`需要的初始学习率较高）。
* 由于`BERT`/`BERT-wwm`使用了维基百科数据进行训练，故它们对正式文本建模较好；而`ERNIE`使用了额外的百度贴吧、知道等网络数据，它对非正式文本（例如微博等）建模有优势。
* 在长文本建模任务上，例如阅读理解、文档分类，`BERT`和`BERT-wwm`的效果较好。
* 如果目标任务的数据和预训练模型的领域相差较大，请在自己的数据集上进一步做预训练。
* 如果要处理繁体中文数据，请使用`BERT`或者`BERT-wwm`。因为我们发现`ERNIE`的词表中几乎没有繁体中文。


## FAQ
**Q: 这个模型怎么用？**  
A: 谷歌发布的中文BERT怎么用，这个就怎么用。
**文本不需要经过分词，wwm只影响预训练过程，不影响下游任务的输入。**

**Q: 请问有预训练代码提供吗？**  
A: 很遗憾，我不能提供相关代码，实现可以参考 [#10](https://github.com/ymcui/Chinese-BERT-wwm/issues/10) 和 [#13](https://github.com/ymcui/Chinese-BERT-wwm/issues/13)。

**Q: 某某数据集在哪里下载？**  
A: 请查看`data`目录，任务目录下的`README.md`标明了数据来源。对于有版权的内容，请自行搜索或与原作者联系获取数据。

**Q: 会有计划发布更大模型吗？比如BERT-large-wwm版本？**  
A: 如果我们从实验中得到更好效果，会考虑发布更大的版本。

**Q: 你骗人！无法复现结果😂**  
A: 在下游任务中，我们采用了最简单的模型。比如分类任务，我们直接使用的是`run_classifier.py`（谷歌提供）。
如果无法达到平均值，说明实验本身存在bug，请仔细排查。
最高值存在很多随机因素，我们无法保证能够达到最高值。
另外一个公认的因素：降低batch size会显著降低实验效果，具体可参考BERT，XLNet目录的相关Issue。

**Q: 我训出来比你更好的结果！**  
A: 恭喜你。

**Q: 训练花了多长时间，在什么设备上训练的？**  
A: 训练是在谷歌TPU v3版本（128G HBM）完成的，训练BERT-wwm花费约1.5天，BERT-wwm-ext则需要数周时间（使用了更多数据需要迭代更充分）。
需要注意的是，预训练阶段我们使用的是`LAMB Optimizer`（[TensorFlow版本实现](https://github.com/ymcui/LAMB_Optimizer_TF)）。该优化器对大的batch有良好的支持。
在微调下游任务时，我们采用的是BERT默认的`AdamWeightDecayOptimizer`。

**Q: ERNIE是谁？**  
A: 本项目中的ERNIE模型特指百度公司提出的[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)，而非清华大学在ACL 2019上发表的[ERNIE](https://github.com/thunlp/ERNIE)。

**Q: BERT-wwm的效果不是在所有任务都很好**  
A: 本项目的目的是为研究者提供多元化的预训练模型，自由选择BERT，ERNIE，或者是BERT-wwm。
我们仅提供实验数据，具体效果如何还是得在自己的任务中不断尝试才能得出结论。
多一个模型，多一种选择。

**Q: 为什么有些数据集上没有试？**  
A: 很坦率的说：
1）没精力找更多的数据；
2）没有必要； 
3）没有钞票；

**Q: 简单评价一下这几个模型**  
A: 各有侧重，各有千秋。
中文自然语言处理的研究发展需要多方共同努力。

**Q: 你预测下一个预训练模型叫什么？**  
A: 可能叫ZOE吧，ZOE: Zero-shOt Embeddings from language model

**Q: 更多关于`RoBERTa-wwm-ext`模型的细节？**  
A: 我们集成了RoBERTa和BERT-wwm的优点，对两者进行了一个自然的结合。
和之前本目录中的模型之间的区别如下:  
1）预训练阶段采用wwm策略进行mask（但没有使用dynamic masking）  
2）简单取消Next Sentence Prediction（NSP）loss  
3）不再采用先max_len=128然后再max_len=512的训练模式，直接训练max_len=512  
4）训练步数适当延长  

需要注意的是，该模型并非原版RoBERTa模型，只是按照类似RoBERTa训练方式训练出的BERT模型，即RoBERTa-like BERT。
故在下游任务使用、模型转换时请按BERT的方式处理，而非RoBERTa。


## 引用
如果本项目中的资源或技术对你的研究工作有所帮助，欢迎在论文中引用下述论文。
- 首选（期刊扩充版）：https://ieeexplore.ieee.org/document/9599397
```
@journal{cui-etal-2021-pretrain,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2021},
  url={https://ieeexplore.ieee.org/document/9599397},
  doi={10.1109/TASLP.2021.3124365},
 }
```

- 或者（会议版本）：https://www.aclweb.org/anthology/2020.findings-emnlp.58
```
@inproceedings{cui-etal-2020-revisiting,
    title = "Revisiting Pre-Trained Models for {C}hinese Natural Language Processing",
    author = "Cui, Yiming  and
      Che, Wanxiang  and
      Liu, Ting  and
      Qin, Bing  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.58",
    pages = "657--668",
}
```



## 免责声明
**本项目并非谷歌官方发布的Chinese BERT-wwm模型。同时，本项目不是哈工大或科大讯飞的官方产品。**
技术报告中所呈现的实验结果仅表明在特定数据集和超参组合下的表现，并不能代表各个模型的本质。
实验结果可能因随机数种子，计算设备而发生改变。
**该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。**


## 问题反馈
如有问题，请在GitHub Issue中提交。



# cnSchema

我们预先处理数据集，并将其关系类型与实体类型与对齐。cnSchema是面向中文信息处理，利用先进的知识图谱、自然语言处理和机器学习技术，融合结构化与文本数据，支持快速领域知识建模，支持跨数据源、跨领域、跨语言的开放数据自动化处理，为智能机器人、语义搜索、智能计算等新兴应用市场提供schema层面的支持与服务。

cnSchema基于的原则
* 完全开放，源自schema.org，OpenKG自主发布的Web Schema
* 立足中文，对接世界
* 面向应用，支持开放数据生态
* 社区共识，知识图谱专家指导

# 实体类型

经过对齐后数据集中所拥有的实体类型如下

| 序号  | 实体类型 | 唯一ID| 序号  | 实体类型 | 唯一ID| 
| --- | :--- | :---: | --- | :--- | :---: | 
| 1  | 人物| YAS |  2 | 影视作品| TOJ | 
| 3  | 目| NGS |  4 | 生物| QCV | 
| 5  | Number| OKB |  6 | Date| BQF | 
| 7  | 国家| CAR |  8 | 网站| ZFM | 
| 9  | 网络小说| EMT |  10 | 图书作品| UER | 
| 11  | 歌曲| QEE |  12 | 地点| UFT | 
| 13  | 气候| GJS |  14 | 行政区| SVA | 
| 15  | TEXT| ANO |  16 | 历史人物| KEJ | 
| 17  | 学校| ZDI |  18 | 企业| CAT | 
| 19  | 出版社| GCK |  20 | 书籍| FQK | 
| 21  | 音乐专辑| BAK |  22 | 城市| RET | 
| 23  | 经典| QZP |  24 | 电视综艺| QAQ | 
| 25 | 机构| ZRE |  26 | 作品| TDZ | 
| 27 | 语言| CVC |  28 | 学科专业| PMN | 


在这之上使用[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1OLx5tjEriMyzbv0iv_s9lihtXWIjB6OS)和[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)为基础训练了DeepKE-cnschema(NER)模型。模型所使用的超参数为所给的参数。最终经过训练后可以得到如下表的效果

<table>
	<tr>
		<th>模型</th>
		<th>P</th>
		<th>R</th>
		<th>F1</th>
	</tr>
	<tr>
		<td>chinese-roberta-wwm-ext(micro)</td>
		<td>0.8028</td>
		<td>0.8612</td>
		<td>0.8310</td>
	</tr>
  <tr>
		<td>chinese-roberta-wwm-ext(macro)</td>
		<td>0.6990</td>
		<td>0.7295</td>
		<td>0.7021</td>
	</tr>
  <tr>
		<td>chinese-bert-wwm(micro)</td>
		<td>0.7841</td>
		<td>0.8587</td>
		<td>0.8197</td>
	</tr>
  <tr>
		<td>chinese-bert-wwm(macro)</td>
		<td>0.6921</td>
		<td>0.7393</td>
		<td>0.7078</td>
	</tr>
	
</table>

# 预测

使用者可以直接下载[模型](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)进行预测，步骤如下：

1、只需将下载文件夹命名为`checkpoints`

2、修改[源码](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/tools/preprocess.py)中的get_labels函数，返回的标签为所给`type.txt`中所用到的标签

```python
def get_labels(self):
    return ['O', 'B-YAS', 'I-YAS', 'B-TOJ', 'I-TOJ', 'B-NGS', 'I-NGS', 'B-QCV', 'I-QCV', 'B-OKB', 'I-OKB', 'B-BQF', 'I-BQF', 'B-CAR', 'I-CAR', 'B-ZFM', 'I-ZFM', 'B-EMT', 'I-EMT', 'B-UER', 'I-UER', 'B-QEE', 'I-QEE', 'B-UFT', 'I-UFT', 'B-GJS', 'I-GJS', 'B-SVA', 'I-SVA', 'B-ANO', 'I-ANO', 'B-KEJ', 'I-KEJ', 'B-ZDI', 'I-ZDI', 'B-CAT', 'I-CAT', 'B-GCK', 'I-GCK', 'B-FQK', 'I-FQK', 'B-BAK', 'I-BAK', 'B-RET', 'I-RET', 'B-QZP', 'I-QZP', 'B-QAQ', 'I-QAQ', 'B-ZRE', 'I-ZRE', 'B-TDZ', 'I-TDZ', 'B-CVC', 'I-CVC', 'B-PMN', 'I-PMN', '[CLS]', '[SEP]']
```
3、只需修改 `predict.yaml`中的参数`text`为需要预测的文本

4、进行预测。需要预测的文本及实体对通过终端返回给程序。

```bash
python predict.py
```
# 训练

如果需要使用其他模型进行训练，步骤如下：

1、也可先下载[数据集](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)，将其放入命名为`data`的文件夹中

2、将`conf`文件夹中的`train.yaml`中的`bert_model`修改为指定模型

3、进行训练。
```bash
python run.py
```
# 样例

使用训练好的模型，只需输入句子“《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽”，运行```python predict.py```后可得到结果，结果显示“星空黑夜传奇”实体类型为经过cnschema对齐后的“网络小说”，“起点中文网”为“网站”，“啤酒的罪孽”为“人物。

## 代码部分

修改 `predict.yaml`中的参数`text`为需要预测的文本
```bash
text=“《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽”
```

最终输出结果

```bash
NER句子：
《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽
NER结果：
[('星','B-UER'),('空','I-UER'),('黑','I-UER'),('夜','I-UER'),('传','I-UER'),('奇','I-UER'),('起','B-ZFM'),('点','I-ZFM'),('中','I-ZFM'),('文','I-ZFM'),('网','I-ZFM'),('啤','B-YAS'),('酒','I-YAS'),('的','I-YAS'),('罪','I-YAS'),('孽','I-YAS')]
```
## 演示gif

具体流程如下gif所示：

<img src="demo.gif" />


