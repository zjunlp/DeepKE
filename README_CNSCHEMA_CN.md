<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="pics/logo_cnschema.png" width="400"/></a>
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
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA.md">English</a> | 简体中文 </b>
</p>

<h1 align="center">
    <p>开源中文知识图谱抽取框架开箱即用特别版DeepKE-cnSchema</p>
</h1>

DeepKE 是一个开源的知识图谱抽取与构建工具，支持<b>低资源、长篇章、多模态</b>的知识抽取工具，可以基于<b>PyTorch</b>实现<b>命名实体识别</b>、<b>关系抽取</b>和<b>属性抽取</b>功能。此版本DeepKE-cnSchema为开箱即用版本，用户下载模型即可实现支持cnSchema的实体和关系知识抽取。

---

## 内容导引

| 章节                          | 描述                                |
| --------------------------- | --------------------------------- |
| [简介](#简介)                   | 介绍DeepKE-cnSchema基本原理             |
| [中文模型下载](#中文模型下载)           | 提供了DeepKE-cnSchema的下载地址           |
| [数据集及中文模型效果](#数据集及中文基线系统效果) | 提供了中文数据集以及中文模型效果                  |
| [快速加载](#快速加载)               | 介绍了如何使用DeepKE-cnSchema进行实体识别、关系抽取 |
| [自定义模型](#自定义模型)             | 提供了使用自定义数据训练模型的说明                 |
| [FAQ](#FAQ)                 | 常见问题答疑                            |
| [引用](#引用)                   | 本目录的技术报告                          |

## 简介

DeepKE 是一个开源的知识图谱抽取与构建工具，支持低资源、长篇章、多模态的知识抽取工具，可以基于PyTorch实现命名实体识别、关系抽取和属性抽取功能。同时为初学者提供了详尽的[文档](https://zjunlp.github.io/DeepKE/)，[Google Colab教程](https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing)，[在线演示](http://deepke.zjukg.cn/)和[幻灯片](https://github.com/zjunlp/DeepKE/blob/main/docs/slides/Slides-DeepKE-cn.pdf)。

为促进中文领域的知识图谱构建和方便用户使用，DeepKE提供了预训练好的支持[cnSchema](https://github.com/OpenKG-ORG/cnSchema)的特别版DeepKE-cnSchema，支持开箱即用的中文实体抽取和关系抽取等任务，可抽取50种关系类型和28种实体类型，其中实体类型包含了通用的人物、地点、城市、机构等类型，关系类型包括了常见的祖籍、出生地、国籍、朝代等类型。

## 中文模型下载

对于实体抽取和关系抽取任务分别提供了基于`RoBERTa-wwm-ext, Chinese`和`BERT-wwm, Chinese`训练的模型。

| 模型简称                                        | 功能                     |                                        Google下载                                         |                                   百度网盘下载                                   |
|:------------------------------------------- |:---------------------- |:---------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| **`DeepKE(NER), RoBERTa-wwm-ext, Chinese`** | **实体抽取** | **[PyTorch](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)** |   **[Pytorch（密码u022）](https://pan.baidu.com/s/1hb9XEbK4x5fIyco4DgZZfg)**   |
| **`DeepKE(NER), BERT-wwm, Chinese`**        | **实体抽取** | **[PyTorch](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)** |   **[Pytorch（密码1g0t）](https://pan.baidu.com/s/10TWE1VA2S-SJgmOm8szRxw)**   |
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

#### 实体Schema

| 序号  | 实体类型    | 序号  | 实体类型  |
| --- |:------ | --- |:---- |
| 1   | cns:人物 YAS  | 2   | cns:影视作品 TOJ | 
| 3   | cns:目 NGS        | 4   | cns:生物 QCV   | 
| 5   | cns:Number OKB   | 6   | cns:Date BQF | 
| 7   | cns:国家 CAR       | 8   | cns:网站 ZFM   | 
| 9   | cns:网络小说 EMT     | 10  | cns:图书作品 UER | 
| 11  | cns:歌曲 QEE       | 12  | cns:地点 UFT   | 
| 13  | cns:气候 GJS       | 14  | cns:行政区 SVA  | 
| 15  | cns:TEXT ANO     | 16  | cns:历史人物 KEJ | 
| 17  | cns:学校 ZDI       | 18  | cns:企业 CAT   | 
| 19  | cns:出版社 GCK      | 20  | cns:书籍 FQK   | 
| 21  | cns:音乐专辑 BAK     | 22  | cns:城市 RET   | 
| 23  | cns:景点 QZP       | 24  | cns:电视综艺 QAQ | 
| 25  | cns:机构 ZRE       | 26  | cns:作品 TDZ   | 
| 27  | cns:语言 CVC       | 28  | cns:学科专业 PMN | 

#### 关系Schema

| 序号  | 头实体类型  | 尾实体类型 | 关系   | 序号  | 头实体类型  | 尾实体类型 | 关系   |
| --- |:------ |:-----:| ---- | --- |:------ |:-----:| ---- |
| 1   | cns:地点     | cns:人物    | cns:祖籍   | 2   | cns:人物     | cns:人物    | cns:父亲   |
| 3   | cns:地点     | cns:企业    | cns:总部地点 | 4   | cns:地点     | cns:人物    | cns:出生地  |
| 5   | cns:目      | cns:生物    | cns:目    | 6   | cns:Number | cns:行政区   | cns:面积   |
| 7   | cns:Text   | cns:机构    | cns:简称   | 8   | cns:Date   | cns:影视作品  | cns:上映时间 |
| 9   | cns:人物     | cns:人物    | cns:妻子   | 10  | cns:音乐专辑   | cns:歌曲    | cns:所属专辑 |
| 11  | cns:Number | cns:企业    | cns:注册资本 | 12  | cns:城市     | cns:国家    | cns:首都   |
| 13  | cns:人物     | cns:影视作品  | cns:导演   | 14  | cns:Text   | cns:历史人物  | cns:字    |
| 15  | cns:Number | cns:人物    | cns:身高   | 16  | cns:企业     | cns:影视作品  | cns:出品公司 |
| 17  | cns:Number | cns:学科专业  | cns:修业年限 | 18  | cns:Date   | cns:人物    | cns:出生日期 |
| 19  | cns:人物     | cns:影视作品  | cns:制片人  | 20  | cns:人物     | cns:人物    | cns:母亲   |
| 21  | cns:人物     | cns:影视作品  | cns:编辑   | 22  | cns:国家     | cns:人物    | cns:国籍   |
| 23  | cns:人物     | cns:影视作品  | cns:编剧   | 24  | cns:网站     | cns:网站小说  | cns:连载网络 |
| 25  | cns:人物     | cns:人物    | cns:丈夫   | 26  | cns:Text   | cns:历史人物  | cns:朝代   |
| 27  | cns:Text   | cns:人物    | cns:民族   | 28  | cns:Text   | cns:历史人物  | cns:朝代   |
| 29  | cns:出版社    | cns:书籍    | cns:出版社  | 30  | cns:人物     | cns:电视综艺  | cns:主持人  |
| 31  | cns:Text   | cns:学科专业  | cns:专业代码 | 32  | cns:人物     | cns:歌曲    | cns:歌手   |
| 33  | cns:人物     | cns:歌曲    | cns:作曲   | 34  | cns:人物     | cns:网络小说  | cns:主角   |
| 35  | cns:人物     | cns:企业    | cns:董事长  | 36  | cns:Date   | cns:企业    | cns:成立时间 |
| 37  | cns:学校     | cns:人物    | cns:毕业院校 | 38  | cns:Number | cns:机构    | cns:占地面积 |
| 39  | cns:语言     | cns:国家    | cns:官方语言 | 40  | cns:Text   | cns:行政区   | cns:人口数量 |
| 41  | cns:Number | cns:行政区   | cns:人口数量 | 42  | cns:城市     | cns:景点    | cns:所在城市 |
| 43  | cns:人物     | cns:图书作品  | cns:作者   | 44  | None   | None    | 其他 |
| 45  | cns:人物     | cns:歌曲    | cns:作曲   | 46  | cns:人物     | cns:行政区   | cns:气候   |
| 47  | cns:人物     | cns:电视综艺  | cns:嘉宾   | 48  | cns:人物     | cns:影视作品  | cns:主演   |
| 49  | cns:作品     | cns:影视作品  | cns:改编自  | 50  | cns:人物     | cns:企业    | cns:创始人  |

## 快速加载

### [实体识别（NER）](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard)

用户可以直接下载[模型](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)进行使用，具体流程如下：

1. 将下载的模型文件夹命名为`checkpoints`
2. 修改 `predict.yaml`中的参数`text`为需要预测的文本

    使用训练好的模型，只需输入句子“《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽”，运行```python predict.py```后可得到结果，结果显示“星空黑夜传奇”实体类型为经过cnschema对齐后的“网络小说”，“起点中文网”为“网站”，“啤酒的罪孽”为“人物。

    ```bash
    text=“《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽”
    ```
3. 预测
    ```shell
    python predict.py
    ```

    最终输出结果

    ```bash
    NER句子：
    《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽
    NER结果：
    [('星','B-UER'),('空','I-UER'),('黑','I-UER'),('夜','I-UER'),('传','I-UER'),('奇','I-UER'),('起','B-ZFM'),('点','I-ZFM'),('中','I-ZFM'),('文','I-ZFM'),('网','I-ZFM'),('啤','B-YAS'),('酒','I-YAS'),('的','I-YAS'),('罪','I-YAS'),('孽','I-YAS')]
    ```

### [关系抽取（RE）](https://github.com/zjunlp/DeepKE/tree/main/example/re/standard)

使用者可以直接下载[模型](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)使用,步骤如下：

1. 修改 `example/re/standard/conf/predict.yaml`中的参数`fp`为下载文件的路径，`example/re/standard/conf/embedding.yaml`中`num_relations`为51（关系个数）,`example/re/standard/conf/config.yaml`中的参数model为`lm`。
    > 注：如无特殊需求，无需修改`example/re/standard/conf/model/lm.yaml`文件
2. 进行预测。需要预测的文本及实体对通过终端输入给程序。

    ```bash
    python predict.py
    ```

    使用训练好的模型，运行```python predict.py```后，只需输入的句子为“歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版”，给定的实体对为“男人的爱”和“人生长路”，可得到结果，最终抽取出的关系为经过cnschema对齐后的“所属专辑”。

    将predict.py的_get_predict_instance函数修改成如下范例，即可修改文本进行预测

    ```python
    def _get_predict_instance(cfg):
        flag = input('是否使用范例[y/n]，退出请输入: exit .... ')
        flag = flag.strip().lower()
        if flag == 'y' or flag == 'yes':
            sentence = '歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版'
            head = '男人的爱'
            tail = '人生长路'
            head_type = '所属专辑'
            tail_type = '歌曲'
        elif flag == 'n' or flag == 'no':
            sentence = input('请输入句子：')
            head = input('请输入句中需要预测关系的头实体：')
            head_type = input('请输入头实体类型：')
            tail = input('请输入句中需要预测关系的尾实体：')
            tail_type = input('请输入尾实体类型：')
        elif flag == 'exit':
            sys.exit(0)
        else:
            print('please input yes or no, or exit!')
            _get_predict_instance()

        instance = dict()
        instance['sentence'] = sentence.strip()
        instance['head'] = head.strip()
        instance['tail'] = tail.strip()
        if head_type.strip() == '' or tail_type.strip() == '':
            cfg.replace_entity_with_type = False
            instance['head_type'] = 'None'
            instance['tail_type'] = 'None'
        else:
            instance['head_type'] = head_type.strip()
            instance['tail_type'] = tail_type.strip()

        return instance
    ```

    最终输出结果

    ```bash
    “男人的爱”和“人生长路”在句中关系为“所属专辑”，置信度为0.99
    ```
    > 注：运行过程中会自动从huggingface网站中下载`example/re/standard/conf/model/lm.yaml`文件中指定的模型，如下载失败可以寻找对应镜像网站或者手动下载

### [联合三元组抽取](https://github.com/zjunlp/DeepKE/tree/main/example/triple)
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


## 自定义模型

### 实体识别任务（NER）

如果需要使用自定义的数据进行训练，步骤如下：

1. 下载自定义的[数据集](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)，将其放入命名为`data`的文件夹中
2. 将`conf`文件夹中的`train.yaml`中的`bert_model`修改为指定模型，用户可以通过修改yaml文件选择不同的模型进行训练（推荐直接下载模型，设置`bert_model`为模型路径）
3. 修改`train.yaml`中的`labels`为`data/type.txt`中所用到的标签
4. 进行训练

    ```bash
    python run.py
    ```

### 关系抽取任务（RE）

如果需要使用其他模型进行训练，步骤如下：

1. 下载自定义的[数据集](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)，将其重命名为`data`
2. 将`conf`文件夹中的`train.yaml`为`lm`,`lm.yaml`中的`lm_file`修改为指定预训练模型，`embedding.yaml`中`num_relations`为关系的个数如51，用户可以通过修改yaml文件选择不同的模型进行训练
3. 进行训练。

    ```bash
    python run.py
    ```

## FAQ

**Q: 这个模型怎么用？**
A: 开箱即用，下载好模型按照使用说明就能够抽取预定义cnSchema包含的知识。
**如果想抽取cnSchema之外的知识，可以使用高级版本自定义数据进行训练哦**

**Q: 请问有其他cnSchema抽取模型提供吗？**
A: 很遗憾，我们暂时只能支持部分cnSchema的知识抽取，未来会发布更多的知识抽取模型。

**Q: 我训出来比你更好的结果！**
A: 恭喜你。

**Q: 自己数据输入进去编码报错**
A: 可能中文输入数据包含了不可见的特殊字符，这些字符无法被某些编码因而报错，您可以通过编辑器或其他工具预处理中文数据解决这一问题。

## 引用

如果本项目中的资源或技术对你的研究工作有所帮助，欢迎在论文中引用下述论文。

```
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

