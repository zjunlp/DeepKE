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
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA_CN.md">简体中文</a> </b>
</p>

<h1 align="center">
    <p>Off-the-shelf Special Edition for Chinese Knowledge Extraction Toolkit——DeepKE-cnSchema</p>
</h1>

DeepKE is a knowledge extraction toolkit based on PyTorch,  supporting **low-resource**, **document-level** and **multimodal** scenarios for **entity**, **relation** and **attribute** extraction. DeepKE-cnSchema is an off-the-shelf version. Users can download the model to realize entity and relation knowledge extraction directly without training.

---

## Catalogue

| Chapter                                                                              | Description                                                                                |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [Introduction](#Introduction)                                                           | Introduce the basic principles of DeepKE-cnSchema                                          |
| [Chinese Model Download](#Chinese-Model-Download)                                       | Provide the download address of DeepKE-cnSchema                                            |
| [Datasets and Chinese Baseline Performance](#Datasets-and-Chinese-Baseline-Performance) | Report the performance of Chinese models                                     |
| [Quick Load](#Quick-Load)                                                               | Introduce how to use DeepKE-cnSchema for entity and relation extraction |
| [User-defined Model](#User-defined-Model)                                               | Provide instructions for training models with customized datasets                           |
| [FAQ](#FAQ)                                                                             | FAQ                                                                                        |
| [Citation](#Citation)                                                                   | Technical report of this catalogue                                                         |

## Introduction

DeepKE is a knowledge extraction toolkit supporting **low-resource**, **document-level** and **multimodal** scenarios for *entity*, *relation* and *attribute* extraction. We provide [documents](https://zjunlp.github.io/DeepKE/), [Google Colab tutorials](https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing), [online demo](http://deepke.zjukg.cn/), and [slides](https://github.com/zjunlp/DeepKE/blob/main/docs/slides/Slides-DeepKE-en.pdf) for beginners.

To promote efficient Chinese knowledge graph construction, we provide DeepKE-cnSchema, a specific version of DeepKE, containing off-the-shelf models based on [cnSchema](https://github.com/OpenKG-ORG/cnSchema). DeepKE-cnSchema supports multiple tasks such as Chinese entity extraction and relation extraction. It can extract 50 relation types and 28 entity types, including common entity types such as person, location, city, institution, etc and the common relation types such as ancestral home, birthplace, nationality and other types.

## Chinese Model Download

For entity extraction and relation extraction tasks, we provide models based on `RoBERTa-wwm-ext, Chinese` and `BERT-wwm, Chinese` respectively.

| Model                                               | Task                          |                                     Google Download                                     |                              Baidu Netdisk Download                               |
| :-------------------------------------------------- | :---------------------------- |:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| **`DeepKE(NER), RoBERTa-wwm-ext, Chinese`** | **entity extraction**   | **[PyTorch](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)** |   **[Pytorch（password:u022）](https://pan.baidu.com/s/1hb9XEbK4x5fIyco4DgZZfg)**   |
| **`DeepKE(NER), BERT-wwm, Chinese`**        | **entity extraction**   | **[PyTorch](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)** |   **[Pytorch（password:1g0t）](https://pan.baidu.com/s/10TWE1VA2S-SJgmOm8szRxw)**   |
| **`DeepKE(NER), BiLSTM-CRF, Chinese`**      | **entity extraction** | **[PyTorch](https://drive.google.com/drive/folders/1n1tzvl6hZYoUUFFWLfkuhkXPx5JB4XK_)** |   **[Pytorch（password:my4x）](https://pan.baidu.com/s/1a9ZFFZVQUxmlbLmbVBaTqQ)**   |
| **`DeepKE(RE), RoBERTa-wwm-ext, Chinese`**  | **relation extraction** | **[PyTorch](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)** |   **[Pytorch（password:78pq）](https://pan.baidu.com/s/1ozFsxExAQTBRs5NbJW7W5g)**   |
| **`DeepKE(RE), BERT-wwm, Chinese`**         | **relation extraction** | **[PyTorch](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)** |   **[Pytorch（password:6psm）](https://pan.baidu.com/s/1ngvTwg_ZXaenxhOeadWoCA)**   |

### Instructions

It is recommended to use Baidu Netdisk download in Chinese Mainland, and Google download for overseas users.

As for the entity extraction model, take pytoch version `DeepKE(RE), RoBERTa-wwm-ext, Chinese` as an example. After downloading, files of the model are obtained:

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

where `config.json` and `vocab.txt` is completely consistent with the original Google `RoBERTa-wwm-ext, Chinese`. PyTorch version contains `pytorch_model. bin`, `config. json`, `vocab. txt` file.

As for the relation extraction model, take pytoch version `DeepKE(RE), RoBERTa-wwm-ext, Chinese` as an example. The model is pth file after downloading.

**After downloading the model, users can directly [quick-load](#Quick-Load) it to extract entity and relation.**

## Datasets and Chinese Baseline Performance

### Datasets

We have conduct experiments on Chinese named entity recognition and relation extraction datasets. The experimental results are as follows:

### Named Entity Recognition(NER)

DeepKE leverages[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1OLx5tjEriMyzbv0iv_s9lihtXWIjB6OS)and[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)to train and obtain the DeepKE-cnSchema(NER) model. Hyper-parameters used in the model are predefined. Finally, we can obtain the following results after training:

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

DeepKE leverages[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)and[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)to train and obtain the DeepKE-cnschema(RE) model. Hyper-parameters used in the model are predefined. Finally, we can obtain the following results after  training:

<table>
    <tr>
        <th>Model</th>
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

### Support Knowledge Schema Type

DeepKE-cnSchema is an off-the-shelf version that supports the Chinese knowledge graphs construction. [CnSchema](https://github.com/OpenKG-ORG/cnSchema) is developed for Chinese information processing, which uses advanced knowledge graphs, natural language processing and machine learning technologies. It integrates structured text data, supports rapid domain knowledge modeling and open data automatic processing across data sources, domains and languages, and provides schema-level support and services for emerging application markets such as intelligent robots, semantic search and intelligent computing. Currently, the Schema types supported by DeepKE-cnSchema are as follows:

#### Entity Schema

| ID | Entity Type | ID | Entity Type | 
| ------------- | :---------- | ------------- | :---------- | 
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

#### Relation Schema

| ID | Head Entity Type | Tail Entity Type | Relation | ID | Head Entity Type | Tail Entity Type | Relation |
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


## Quick Load

### [Named Entity Recognition(NER)](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard)

Users can directly download the [model](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg) for usage. The details are as follows：

1. Create the downloaded folder as `checkpoints`
2. Set the parameter `text` in `predict.yaml` as the sentence to be predicted
	To use the trained model, just set the input sentence "《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽". After running `python oredict.py`, results can be obtained which show that the entity type "星空黑夜传奇" is "网络小说", "起点中文网" is "网站" and "啤酒的罪孽" is "人物".

	```bash
	text="《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽"
	```
3. Predict
	```bash
	python predict.py
	```


	Finally, output the results:

	```bash
	NER句子：
	《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽
	NER结果：
	[('星','B-UER'),('空','I-UER'),('黑','I-UER'),('夜','I-UER'),('传','I-UER'),('奇','I-UER'),('起','B-ZFM'),('点','I-ZFM'),('中','I-ZFM'),('文','I-ZFM'),('网','I-ZFM'),('啤','B-YAS'),('酒','I-YAS'),('的','I-YAS'),('罪','I-YAS'),('孽','I-YAS')]
	```

### [Relation Extraction(RE)](https://github.com/zjunlp/DeepKE/tree/main/example/re/standard)
Users can directly download the [model](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv) for usage. The details are as follows：

1. Modify the parameter `fp`in `example/re/standard/conf/config.yaml`to the path of downloaded file, `num_relations`in `example/re/standard/conf/embedding.yaml`to 51(relation nums) and `model` in `example/re/standard/conf/config.yaml`to lm.
	> Note: There is no need to modify the contents of `example/re/standard/conf/model/lm.yaml` unless there are specific requirements.
2. Predict. The text and entity pairs to be predicted are fed to the program through the terminal.

	```bash
	python predict.py
	```

	To use the trained model, run `python predict.py` and input the sentence "歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版". The given entity pair are "男人的爱" and "人生长路". Finally, the extracted relation is "所属专辑".

	To change the text to be predicted, modify the `_get_predict_instance`function in `predict.py` to the following example:

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
 	> Note: The model specified in `example/re/standard/conf/model/lm.yaml` will be automatically downloaded from the huggingface website during runtime. If the download fails, please use the huggingface mirror site or download it manually.

### [Joint Entity and Relation Extraction](https://github.com/zjunlp/DeepKE/tree/main/example/triple)
After aforementioned trained models are downloaded, entites and their relations in a text can be extracted together. If there are more than two entities in one sentence, some predicted entity pairs may be incorrect because these entity pairs are not in training sets and need to be exracted further. The detailed steps are as follows:<br>
1. In `conf`, modify `text` in `predict.yaml` as the sentence to be predicted, `nerfp` as the directory of the trained NER model and `refp` as the directory of the trained RE model.
2. Predict
	```shell
	python predict.py
	```
	Many results will be output. Take the input text `此外网易云平台还上架了一系列歌曲，其中包括田馥甄的《小幸运》等` as example.
	
	(1) Output the result of NER: `[('田', 'B-YAS'), ('馥', 'I-YAS'), ('甄', 'I-YAS'), ('小', 'B-QEE'), ('幸', 'I-QEE'), ('运', 'I-QEE')]`
	
	(2) Output the processed result: `{'田馥甄': '人物', '小幸运': '歌曲'}`
	
	(3) Output the result of RE: `"田馥甄" 和 "小幸运" 在句中关系为："歌手"，置信度为0.92。`
	
	(4) Output the result as `jsonld`
	
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

## Custom Models (Advanced Usage)

### Named Entity Recognition (NER)

If you need to use customized dataset for training, follow the steps bellow:

1. Download customized [dataset](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg) and put it into the `data` folder.
2. Modify the parameter `bert_model`in `train.yaml`of the `conf`folder to the specify model. Users can choose different models to train by modifying the `yaml`file.
3. Modify `labels` in `train.yaml` as the labels in `data/type.txt`
4. Train.
	```bash
	python run.py
	```

### Relation Extraction (RE)

If you need to use other models for training, follow the steps bellow:

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

## Disclaimers

**The contents of this project are only for technical research reference and shall not be used as any conclusive basis. Users can freely use the model within the scope of the license, but we are not responsible for the direct or indirect losses caused by the use of the project.**

## Problem Feedback

If you have any questions, please submit them in GitHub issue.

