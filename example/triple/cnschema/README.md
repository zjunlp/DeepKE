<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="../../../pics/logo_cnschema.png" width="400"/></a>


<b> English | <a href="./README_CN.md">简体中文</a>  </b>



## Introduction

DeepKE is a knowledge extraction toolkit supporting **low-resource**, **document-level** and **multimodal** scenarios for *entity*, *relation* and *attribute* extraction. We provide [documents](https://zjunlp.github.io/DeepKE/), [Google Colab tutorials](https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing), [online demo](http://deepke.zjukg.cn/), and [slides](https://github.com/zjunlp/DeepKE/blob/main/docs/slides/Slides-DeepKE-en.pdf) for beginners.

To promote efficient Chinese knowledge graph construction, we provide DeepKE-cnSchema, a specific version of DeepKE, containing off-the-shelf models based on [cnSchema](https://github.com/OpenKG-ORG/cnSchema). DeepKE-cnSchema supports multiple tasks such as Chinese entity extraction and relation extraction. It can extract 50 relation types and 28 entity types, including common entity types such as person, location, city, institution, etc and the common relation types such as ancestral home, birthplace, nationality and other types.

## Chinese Model Download

For entity extraction and relation extraction tasks, we provide models based on `RoBERTa-wwm-ext, Chinese` and `BERT-wwm, Chinese` respectively.

| Model                                               | Task                          |                                     Google Download                                     |                              Baidu Netdisk Download                               |
| :-------------------------------------------------- | :---------------------------- |:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| **`DeepKE(NER), RoBERTa-wwm-ext, Chinese`** | **entity extraction**   | **[PyTorch](https://drive.google.com/drive/folders/1T3xf_MXRaVqLV-ST4VqvKoaQqQgRpp67)** |   **[Pytorch（password:u022）](https://pan.baidu.com/s/1hb9XEbK4x5fIyco4DgZZfg)**   |
| **`DeepKE(NER), BERT-wwm, Chinese`**        | **entity extraction**   | **[PyTorch](https://drive.google.com/drive/folders/1OLx5tjEriMyzbv0iv_s9lihtXWIjB6OS)** |   **[Pytorch（password:1g0t）](https://pan.baidu.com/s/10TWE1VA2S-SJgmOm8szRxw)**   |
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


## Quick Load
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


## Citation

If the resources or technologies in this project are helpful to your research work, you are welcome to cite the following papers in your thesis:

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

## Disclaimers

**The contents of this project are only for technical research reference and shall not be used as any conclusive basis. Users can freely use the model within the scope of the license, but we are not responsible for the direct or indirect losses caused by the use of the project.**

## Problem Feedback

If you have any questions, please submit them in GitHub issue.
