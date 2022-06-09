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

<h1 align="center">
    <p>Off-the-shelf Special Edition for Chinese Knowledge Extraction Toolkit——DeepKE-cnSchema</p>
</h1>

DeepKE is a knowledge extraction toolkit based on PyTorch,  supporting **low-resource**, **document-level** and **multimodal** scenarios for **entity**, **relation** and **attribute** extraction. DeepKE-cnSchema is an off-the-shelf version. Users can download the model to realize entity and relation knowledge extraction directly which supports cnSchema.

---

## Catalogue

| Chapter                                                                              | Description                                                                                |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [Introduction](#Introduction)                                                           | Introduce the basic principles of DeepKE-cnSchema                                          |
| [Chinese Model Download](#Chinese-Model-Download)                                       | Provide the download address of DeepKE-cnSchema                                            |
| [Datasets and Chinese Baseline Performance](#Datasets-and-Chinese-Baseline-Performance) | Provide Chinese datasets and Chinese model performance                                     |
| [Quick Load](#Quick-Load)                                                               | Introduce how to use DeepKE-cnSchema to realize entity recognition and relation extraction |
| [User-defined Model](#User-defined-Model)                                               | Provide instructions for training models with customized dataset                           |
| [FAQ](#FAQ)                                                                             | FAQ                                                                                        |
| [Citation](#Citation)                                                                   | Technical report of this catalogue                                                         |

## Introduction

DeepKE is a knowledge extraction toolkit supporting **low-resource**, **document-level** and **multimodal** scenarios for *entity*, *relation* and *attribute* extraction. We provide [comprehensive documents](https://zjunlp.github.io/DeepKE/), [Google Colab tutorials](https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing), and [online demo](http://deepke.zjukg.cn/) for beginners.

In order to promote the Chinese knowledge graph construction and make it user friendly, we provide DeepKE-cnSchema, a special version of DeepKE, containing pretrained models which support [cnSchema](https://github.com/OpenKG-ORG/cnSchema). DeepKE-cnSchema supports off-the-shelf tasks such as Chinese entity extraction and relation extraction. It can extract 50 relation types and 28 entity types, of which the entity types are common ones such as person, location, city, institution, etc and the relation types include ancestral home, birthplace, nationality and other types.

## Chinese Model Download

For entity extraction and relation extraction tasks, we provide models based on `RoBERTa-wwm-ext, Chinese` and `BERT-wwm, Chinese` respectively.

| Model                                               | Task                          |                                   Google Download                                   |                               Baidu Netdisk Download                               |
| :-------------------------------------------------- | :---------------------------- | :----------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| **`DeepKE(NER), RoBERTa-wwm-ext, Chinese`** | **entity extraction**   | **[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[Pytorch（password:u022）](https://pan.baidu.com/s/1hb9XEbK4x5fIyco4DgZZfg)** |
| **`DeepKE(NER), BERT-wwm, Chinese`**        | **entity extraction**   | **[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[Pytorch（password:nmiv）](https://pan.baidu.com/s/1oi2K6vtOr8b87FkCTIkQsA)** |
| **`DeepKE(RE), RoBERTa-wwm-ext, Chinese`**  | **relation extraction** | **[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[Pytorch（password:29oe）](https://pan.baidu.com/s/1kPoihfHzVtxKLavUMCDLJw)** |
| **`DeepKE(RE), BERT-wwm, Chinese`**         | **relation extraction** | **[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[Pytorch（password:e7e9）](https://pan.baidu.com/s/1pLzOizjBgVT-GD1yNvn8dg)** |

### Instructions

It is recommended to use Baidu Netdisk download in Chinese Mainland, and Google download for overseas users.

As for the entity extraction model, take pytoch version `DeepKE(RE), RoBERTa-wwm-ext, Chinese` as an example. After downloading, unzip the zip file to get:

```
checkpoints_robert
    |- added_tokens.json          # added tokens
    |- config.json                # config
    |- eval_results.txt           # evaluation results
    |- model_config.json          # model config
    |- pytorch_model.bin          # model
    |- special_tokens_map.json    # special tokens map
    |- tokenizer_config.bin       # tokenizer config
    |- vocab.txt                  # vocabulary
```

where `config.json` and `vocab.txt` is completely consistent with the original Google `RoBERTa-wwm-ext, Chinese`. PyTorch version contains `pytorch_ model. bin`, `config. json`, `vocab. txt` file.

As for the relation extraction model, take pytoch version `DeepKE(RE), RoBERTa-wwm-ext, Chinese` as an example. The model is pth file after downloading.

**After downloading the model, users can directly [quick-load](#Quick-Load) it to extract entity and relation.**

## Datasets and Chinese Baseline Performance

### Datasets

We have carried out experiments on Chinese named entity recognition and relation extraction datasets. The experimental results are as follows:

### Named Entity Recognition(NER)

DeepKE leverages[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1OLx5tjEriMyzbv0iv_s9lihtXWIjB6OS)and[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)to train and obtain the DeepKE-cnSchema(NER) model. Hyper-parameters used in the model are predifined. Finally, we can get the following results after  training:

<table>
    <tr>
        <th>Model</th>
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

### Relation Extraction(RE)

DeepKE leverages[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)and[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)to train and obtain the DeepKE-cnschema(RE) model. Hyper-parameters used in the model are predefined. Finally, we can get the following results after  training:

<table>
    <tr>
        <th>Model</th>
        <th>P</th>
        <th>R</th>
        <th>F1</th>
    </tr>
  <tr>
        <td><b>DeepKE(RE), RoBERTa-wwm-ext, Chinese</b></td>
        <td>0.8761</td>
        <td>0.8598</td>
        <td>0.8665</td>
    </tr>
  <tr>
        <td><b>DeepKE(RE), BERT-wwm, Chinese</b></td>
        <td>0.8742</td>
        <td>0.8582</td>
        <td>0.8639</td>
    </tr>
</table>

### Support Knowledge Schema Type

DeepKE-cnSchema is an off-the-shelf version that supports the Chinese knowledge graphs construction. [Cnschema](https://github.com/OpenKG-ORG/cnSchema) is oriented to Chinese information processing which uses advanced knowledge graphs, natural language processing and machine learning technologies, integrates structured text data, supports rapid domain knowledge modeling, supports open data automatic processing across data sources, domains and languages, and provides schema level support and services for emerging application markets such as intelligent robots, semantic search and intelligent computing. Currently, the Schema types supported by Deepke-cnSchema are as follows:

#### Entity Schema

| Serial Number | Entity Type | ID | Serial Number | Entity Type | ID |
| ------------- | :---------- | :-: | ------------- | :---------- | :-: |
| 1             | 人物        | YAS | 2             | 影视作品    | TOJ |
| 3             | 目          | NGS | 4             | 生物        | QCV |
| 5             | Number      | OKB | 6             | Date        | BQF |
| 7             | 国家        | CAR | 8             | 网站        | ZFM |
| 9             | 网络小说    | EMT | 10            | 图书作品    | UER |
| 11            | 歌曲        | QEE | 12            | 地点        | UFT |
| 13            | 气候        | GJS | 14            | 行政区      | SVA |
| 15            | TEXT        | ANO | 16            | 历史人物    | KEJ |
| 17            | 学校        | ZDI | 18            | 企业        | CAT |
| 19            | 出版社      | GCK | 20            | 书籍        | FQK |
| 21            | 音乐专辑    | BAK | 22            | 城市        | RET |
| 23            | 经典        | QZP | 24            | 电视综艺    | QAQ |
| 25            | 机构        | ZRE | 26            | 作品        | TDZ |
| 27            | 语言        | CVC | 28            | 学科专业    | PMN |

#### Relation Schema

| Serial Number | Head Entity Type | Tail Entity Type | Relation | Serial Number | Head Entity Type | Tail Entity Type | Relation |
| ------------- | :--------------- | :--------------: | -------- | ------------- | :--------------- | :--------------: | -------- |
| 1             | 地点             |       人物       | 祖籍     | 2             | 人物             |       人物       | 父亲     |
| 3             | 地点             |       企业       | 总部地点 | 4             | 地点             |       人物       | 出生地   |
| 5             | 目               |       生物       | 目       | 6             | Number           |      行政区      | 面积     |
| 7             | Text             |       机构       | 简称     | 8             | Date             |     影视作品     | 上映时间 |
| 9             | 人物             |       人物       | 妻子     | 10            | 音乐专辑         |       歌曲       | 所属专辑 |
| 11            | Number           |       企业       | 注册资本 | 12            | 城市             |       国家       | 首都     |
| 13            | 人物             |     影视作品     | 导演     | 14            | Text             |     历史人物     | 字       |
| 15            | Number           |       人物       | 身高     | 16            | 企业             |     影视作品     | 出品公司 |
| 17            | Number           |     学科专业     | 修业年限 | 18            | Date             |       人物       | 出生日期 |
| 19            | 人物             |     影视作品     | 制片人   | 20            | 人物             |       人物       | 母亲     |
| 21            | 人物             |     影视作品     | 编辑     | 22            | 国家             |       人物       | 国籍     |
| 23            | 人物             |     影视作品     | 编剧     | 24            | 网站             |     网站小说     | 连载网络 |
| 25            | 人物             |       人物       | 丈夫     | 26            | Text             |     历史人物     | 朝代     |
| 27            | Text             |       人物       | 民族     | 28            | Text             |     历史人物     | 朝代     |
| 29            | 出版社           |       书籍       | 出版社   | 30            | 人物             |     电视综艺     | 主持人   |
| 31            | Text             |     学科专业     | 专业代码 | 32            | 人物             |       歌曲       | 歌手     |
| 33            | 人物             |       歌曲       | 作曲     | 34            | 人物             |     网络小说     | 主角     |
| 35            | 人物             |       企业       | 董事长   | 36            | Date             |       机构       | 成立时间 |
| 37            | 学校             |       人物       | 毕业院校 | 38            | Number           |       机构       | 占地面积 |
| 39            | 语言             |       国家       | 官方语言 | 40            | Text             |      行政区      | 人口数量 |
| 41            | Number           |      行政区      | 人口数量 | 42            | 城市             |       景点       | 所在城市 |
| 43            | 人物             |     图书作品     | 作者     | 44            | Date             |       企业       | 成立时间 |
| 45            | 人物             |       歌曲       | 作曲     | 46            | 人物             |      行政区      | 气候     |
| 47            | 人物             |     电视综艺     | 嘉宾     | 48            | 人物             |     影视作品     | 主演     |
| 49            | 作品             |     影视作品     | 改编自   | 50            | 人物             |       企业       | 创始人   |

## Quick Load

### Named Entity Recognition(NER)

Users can directly download the [model](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg) to use. The details are as follows：

1、Name the downloaded folder as `checkpoints`

2、Modify the `get_labels`function in the [source code](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/tools/preprocess.py). The returned labels are given in `type.txt`

```python
def get_labels(self):
    return ['O', 'B-YAS', 'I-YAS', 'B-TOJ', 'I-TOJ', 'B-NGS', 'I-NGS', 'B-QCV', 'I-QCV', 'B-OKB', 'I-OKB', 'B-BQF', 'I-BQF', 'B-CAR', 'I-CAR', 'B-ZFM', 'I-ZFM', 'B-EMT', 'I-EMT', 'B-UER', 'I-UER', 'B-QEE', 'I-QEE', 'B-UFT', 'I-UFT', 'B-GJS', 'I-GJS', 'B-SVA', 'I-SVA', 'B-ANO', 'I-ANO', 'B-KEJ', 'I-KEJ', 'B-ZDI', 'I-ZDI', 'B-CAT', 'I-CAT', 'B-GCK', 'I-GCK', 'B-FQK', 'I-FQK', 'B-BAK', 'I-BAK', 'B-RET', 'I-RET', 'B-QZP', 'I-QZP', 'B-QAQ', 'I-QAQ', 'B-ZRE', 'I-ZRE', 'B-TDZ', 'I-TDZ', 'B-CVC', 'I-CVC', 'B-PMN', 'I-PMN', '[CLS]', '[SEP]']
```

3、Modify the parameter `text` in `predict.yaml` to the text to be predicted

4、Predict. The text and entity pairs to be predicted are input to the program through the terminal

```bash
python predict.py
```

To use the trained model, just input the sentence "《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽". After running `python oredict.py`, results can be obtained which show that the entity type "星空黑夜传奇" is "网络小说" aligned with cnschema, "起点中文网" is "网站" and "啤酒的罪孽" is "人物".

Modify the parameter `text` in `predict.yaml` to the text to be predicted

```bash
text=“《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽”
```

Finally, output the results:

```bash
NER句子：
《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽
NER结果：
[('星','B-UER'),('空','I-UER'),('黑','I-UER'),('夜','I-UER'),('传','I-UER'),('奇','I-UER'),('起','B-ZFM'),('点','I-ZFM'),('中','I-ZFM'),('文','I-ZFM'),('网','I-ZFM'),('啤','B-YAS'),('酒','I-YAS'),('的','I-YAS'),('罪','I-YAS'),('孽','I-YAS')]
```

### Relation Extraction(RE)

Users can directly download the [model](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv) to use. The details are as follows：

1、Modify the parameter `fp`in `predict.yaml`to the path of downloaded file and `num_relations`in `embedding.yaml`to 51(relation nums)

2、Predict. The text and entity pairs to be predicted are input to the program through the terminal

```bash
python predict.py
```

To use the trained model, run `python predict.py` and input the sentence "歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版". The given entity pair are "男人的爱" and "人生长路". Finally, the extracted relation is "所属专辑" aligned with cnschema.

To change the text to be predicted, modify the `_get_predict_instance`function in `predict.py` to the following example:

```python
def _get_predict_instance(cfg):
    flag = input('是否使用范例[y/n]，退出请输入: exit .... ')
    flag = flag.strip().lower()
    if flag == 'y' or flag == 'yes':
        sentence = '歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版'
        head = '男人的爱'
        tail = '人生长路'
        head_type = ''
        tail_type = ''
    elif flag == 'n' or flag == 'no':
        sentence = input('请输入句子：')
        head = input('请输入句中需要预测关系的头实体：')
        head_type = input('请输入头实体类型（可以为空，按enter跳过）：')
        tail = input('请输入句中需要预测关系的尾实体：')
        tail_type = input('请输入尾实体类型（可以为空，按enter跳过）：')
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

Finally, output the results:

```bash
“男人的爱”和“人生长路”在句中关系为“所属专辑”，置信度为0.99
```

## User-defined Model

### Named Entity Recognition(NER)

If you need to use customized dataset for training, the steps are as follows:

1、Download customized [dataset](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg) and put it into the `data` folder.

2、Modify the parameter `bert_model`in `train.yaml`of the `conf`folder to the specify model. Users can choose different models to train by modifying the `yaml`file.

3、Train.

```bash
python run.py
```

### Relation Extraction(RE)

If you need to use other models for training, the steps are as follows:

1、Download the customized [dataset](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv) and rename it to `data`.

2、Modify the parameter `model_name`in `train.yaml`of the `conf`folder to `lm`, `num_relations`in `embedding_yaml`to the number of relations(eg: 51). Users can choose different models to train by modifying the `yaml`file.

3、Train.

```bash
python run.py
```

## FAQ

**Q: How to use this model?**
A: It is off-the-shelf. After downloading the model, follow the instructions and you can extract the knowledge contained in the predefined cnSchema.
**If you want to extract knowledge other than cnSchema, you can use the advanced version of customized data for training**

**Q: Is there any other cnSchema extraction model available?**
A: Unfortunately, we can only support part of knowledge extraction of cnSchema for the time being. More knowledge extraction models will be published in the future.

**Q: I trained better result than you!**
A: Congratulations!

**Q: Embedding error for customized dataset.**
A: The Chinese data may contain invisible special characters, which cannot be encoded and thus an error is reported. You can preprocess the Chinese data through the editor or other tools to solve this problem.

## Citation

If the resources or technologies in this project are helpful to your research work, you are welcome to cite the following papers in your thesis:

```
@article{zhang2022deepke,
  title={DeepKE: A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population},
  author={Zhang, Ningyu and Xu, Xin and Tao, Liankuan and Yu, Haiyang and Ye, Hongbin and Xie, Xin and Chen, Xiang and Li, Zhoubo and Li, Lei and Liang, Xiaozhuan and others},
  journal={arXiv preprint arXiv:2201.03335},
  year={2022}
}
```

## Disclaimers

**The contents of this project are only for technical research reference and shall not be used as any conclusive basis. Users can freely use the model within the scope of the license, but we are not responsible for the direct or indirect losses caused by the use of the project.**

## Problem Feedback

If you have any questions, please submit them in GitHub issue.

