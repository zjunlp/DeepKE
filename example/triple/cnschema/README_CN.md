<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="../../../pics/logo_cnschema.png" width="400"/></a>
<p>


<b> <a href="./README.md">English</a> | 简体中文 </b>


<h1 align="center">
    <p>开源中文知识图谱抽取框架开箱即用特别版DeepKE-cnSchema</p>
</h1>

---

## 简介

DeepKE 是一个开源的知识图谱抽取与构建工具，支持低资源、长篇章、多模态的知识抽取工具，可以基于PyTorch实现命名实体识别、关系抽取和属性抽取功能。同时为初学者提供了详尽的[文档](https://zjunlp.github.io/DeepKE/)，[Google Colab教程](https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing)，[在线演示](http://deepke.zjukg.cn/)和[幻灯片](https://github.com/zjunlp/DeepKE/blob/main/docs/slides/Slides-DeepKE-cn.pdf)。

为促进中文领域的知识图谱构建和方便用户使用，DeepKE提供了预训练好的支持[cnSchema](https://github.com/OpenKG-ORG/cnSchema)的特别版DeepKE-cnSchema，支持开箱即用的中文实体抽取和关系抽取等任务，可抽取50种关系类型和28种实体类型，其中实体类型包含了通用的人物、地点、城市、机构等类型，关系类型包括了常见的祖籍、出生地、国籍、朝代等类型。

## 中文模型下载

对于实体抽取和关系抽取任务分别提供了基于`RoBERTa-wwm-ext, Chinese`和`BERT-wwm, Chinese`训练的模型。

| 模型简称                                        | 功能                     |                                        Google下载                                         |                                   百度网盘下载                                   |
|:------------------------------------------- |:---------------------- |:---------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| **`DeepKE(NER), RoBERTa-wwm-ext, Chinese`** | **实体抽取** | **[PyTorch](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)** |   **[Pytorch（密码u022）](https://pan.baidu.com/s/1hb9XEbK4x5fIyco4DgZZfg)**   |
| **`DeepKE(NER), BERT-wwm, Chinese`**        | **实体抽取** | **[PyTorch](https://drive.google.com/drive/folders/1OLx5tjEriMyzbv0iv_s9lihtXWIjB6OS)** |   **[Pytorch（密码1g0t）](https://pan.baidu.com/s/10TWE1VA2S-SJgmOm8szRxw)**   |
| **`DeepKE(NER), BiLSTM-CRF, Chinese`**      | **实体抽取** | **[PyTorch](https://drive.google.com/drive/folders/1n1tzvl6hZYoUUFFWLfkuhkXPx5JB4XK_)** |   **[Pytorch（密码my4x）](https://pan.baidu.com/s/1a9ZFFZVQUxmlbLmbVBaTqQ)**   |
| **`DeepKE(RE), RoBERTa-wwm-ext, Chinese`**  | **关系抽取** | **[PyTorch](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)** |   **[Pytorch（密码78pq）](https://pan.baidu.com/s/1ozFsxExAQTBRs5NbJW7W5g)**   |
| **`DeepKE(RE), BERT-wwm, Chinese`**         | **关系抽取** | **[PyTorch](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)** |   **[Pytorch（密码6psm）](https://pan.baidu.com/s/1ngvTwg_ZXaenxhOeadWoCA)**   |

### 使用说明

中国大陆境内建议使用百度网盘下载点，境外用户建议使用谷歌下载点。
实体抽取模型中，以Pytorch版`DeepKE(RE), RoBERTa-wwm-ext, Chinese`为例，下载完毕后得到模型文件：

```
checkpoints_robert
    |- added_tokens.json          # 额外增加词表
    |- config.json                # 整体参数
    |- eval_results.txt           # 验证结果
    |- model_config.json          # 模型参数
    |- pytorch_model.bin          # 模型
    |- special_tokens_map.json    # 特殊词表映射
    |- tokenizer_config.bin       # 分词器参数
    |- vocab.txt                  # 词表
```

其中`config.json`和`vocab.txt`与谷歌原版`RoBERTa-wwm-ext, Chinese`完全一致。
PyTorch版本则包含`pytorch_model.bin`, `config.json`, `vocab.txt`文件。

关系抽取模型中，以Pytorch版`DeepKE(RE), RoBERTa-wwm-ext, Chinese`为例，下载后为pth文件。

**下载模型后，用户即可直接[快速加载](#快速加载)模型进行实体关系抽取。**

## 数据集及中文基线系统效果

### 数据集

我们在中文实体识别和关系抽取数据集上进行了实验，实验结果如下

### 实体识别（NER）

DeepKE使用[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1OLx5tjEriMyzbv0iv_s9lihtXWIjB6OS)和[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)为基础训练得到了DeepKE-cnSchema(NER)模型。模型所使用的超参数均为预定义的参数。最终经过训练后可以得到如下表的效果

<table>
    <tr>
        <th>模型</th>
        <th>P</th>
        <th>R</th>
        <th>F1</th>
    </tr>
    <tr>
        <td><b>DeepKE(NER), RoBERTa-wwm-ext, Chinese</b></td>
        <td>0.8028</td>
        <td>0.8612</td>
        <td>0.8310</td>
    </tr>

<tr>
        <td><b>DeepKE(NER), BERT-wwm, Chinese</b></td>
        <td>0.7841</td>
        <td>0.8587</td>
        <td>0.8197</td>
    </tr>

</table>

### 关系抽取（RE）

DeepKE使用[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)和[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)为基础得到了DeepKE-cnschema(RE)模型。模型所使用的超参数均为预定义的参数。最终经过训练后可以得到如下表的效果

<table>
    <tr>
        <th>模型</th>
        <th>P</th>
        <th>R</th>
        <th>F1</th>
    </tr>
  <tr>
        <td><b>DeepKE(RE), RoBERTa-wwm-ext, Chinese</b></td>
        <td>0.7890</td>
        <td>0.7370</td>
        <td>0.7327</td>
    </tr>
  <tr>
        <td><b>DeepKE(RE), BERT-wwm, Chinese</b></td>
        <td>0.7861</td>
        <td>0.7506</td>
        <td>0.7473</td>
    </tr>

</table>

### 支持知识Schema类型

DeepKE-cnSchema特别版为支持中文领域知识图谱构建推出的开箱即用版本。 [CnSchema](https://github.com/OpenKG-ORG/cnSchema)是面向中文信息处理，利用先进的知识图谱、自然语言处理和机器学习技术，融合结构化与文本数据，支持快速领域知识建模，支持跨数据源、跨领域、跨语言的开放数据自动化处理，为智能机器人、语义搜索、智能计算等新兴应用市场提供schema层面的支持与服务。目前，DeepKE-cnSchema支持的Schema类型如下表所示：



## 快速加载

用户可以先将上述模型下载至本地，然后使用[example/triple](https://github.com/zjunlp/DeepKE/tree/main/example/triple)中的代码进行三元组抽取。如果单句中存在超过两个以上的实体数，可能在一些实体对中会存在预测不准确的问题，那是因为这些实体对并没有被加入训练集中进行训练，所以需要进一步判断，具体使用步骤如下：

1. 将`conf`文件夹中的`predict.yaml`中的`text`修改为预测文本，`nerfp`修改为ner模型文件夹地址，`refp`为re模型地址
2. 进行预测。

    ```bash
    python predict.py
    ```

    期间将输出各个中间步骤结果，以输入文本`此外网易云平台还上架了一系列歌曲，其中包括田馥甄的《小幸运》等`为例。

    2.1 输出经过ner模型后得到结果`[('田', 'B-YAS'), ('馥', 'I-YAS'), ('甄', 'I-YAS'), ('小', 'B-QEE'), ('幸', 'I-QEE'), ('运', 'I-QEE')]`。

    2.2 输出进行处理后结果`{'田馥甄': '人物', '小幸运': '歌曲'}`

    2.3 输出经过re模型后得到结果` "田馥甄" 和 "小幸运" 在句中关系为："歌手"，置信度为0.92。`

    2.4 输出jsonld格式化后结果 
    ```bash
    {
      "@context": {
        "歌手": "https://cnschema.openkg.cn/item/%E6%AD%8C%E6%89%8B/16693#viewPageContent"
      },
      "@id": "田馥甄",
      "歌手": {
        "@id": "小幸运"
      }
    }
    ```


## 引用

如果本项目中的资源或技术对你的研究工作有所帮助，欢迎在论文中引用下述论文。

```bibtex
@inproceedings{DBLP:conf/emnlp/ZhangXTYYQXCLL22,
  author    = {Ningyu Zhang and
               Xin Xu and
               Liankuan Tao and
               Haiyang Yu and
               Hongbin Ye and
               Shuofei Qiao and
               Xin Xie and
               Xiang Chen and
               Zhoubo Li and
               Lei Li},
  editor    = {Wanxiang Che and
               Ekaterina Shutova},
  title     = {DeepKE: {A} Deep Learning Based Knowledge Extraction Toolkit for Knowledge
               Base Population},
  booktitle = {Proceedings of the The 2022 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2022 - System Demonstrations, Abu Dhabi,
               UAE, December 7-11, 2022},
  pages     = {98--108},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.emnlp-demos.10},
  timestamp = {Thu, 23 Mar 2023 16:56:00 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/ZhangXTYYQXCLL22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## 免责声明

**该项目中的内容仅供技术研究参考，不作为任何结论性依据。使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。**

## 问题反馈

如有问题，请在GitHub Issue中提交。

