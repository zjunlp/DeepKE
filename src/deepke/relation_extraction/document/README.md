[**中文**](https://github.com/zjunlp/DeepKE/edit/test_new_deepke/src/deepke/relation_extraction/document/README_CN.md) | [**English**](https://github.com/zjunlp/DeepKE/edit/test_new_deepke/src/deepke/relation_extraction/document/README.md)

>

<p align="center">
  	<font size=7><strong>DocNet:Document-level Relation Extraction as Semantic Segmentation</strong></font>
</p>



This repository is the official implementation of [**DocuNet**](https://github.com/zjunlp/DocRE/), which is model proposed in a paper: **[Document-level Relation Extraction as Semantic Segmentation](https://www.ijcai.org/proceedings/2021/551)**, accepted by **IJCAI2021** main conference. 


# Contributor
Xiang Chen, Xin Xie, Shuming Deng, Ningyu Zhang, and Huajun Chen. 


# Brief Introduction
This paper innovatively proposes the DocuNet model, which first regards the document-level relation extraction as the semantic segmentation task in computer vision.


# Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


# Training

To train the DocuNet model in the paper on the dataset [DocRED](https://github.com/thunlp/DocRE), run this command:

```bash
>> bash scripts/run_docred.sh # use BERT/RoBERTa by setting --transformer-type
```

To train the DocuNet model in the paper on the dataset CDR and GDA, run this command:

```bash
>> bash scripts/run_cdr.sh  # for CDR
>> bash scripts/run_gda.sh  # for GDA
```



# Evaluation

To evaluate the trained model in the paper, you setting the `--load_path` argument in training scripts. The program will log the result of evaluation automatically. And for DocRED  it will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.


# Results

Our model achieves the following performance on : 

### Document-level Relation Extraction on [DocRED](https://github.com/thunlp/DocRED)


| Model     | Ign F1 on Dev | F1 on Dev | Ign F1 on Test | F1 on Test |
| :----------------: |:--------------: | :------------: | ------------------ | ------------------ |
| DocuNet-BERT (base) |  59.86±0.13 |   61.83±0.19 |     59.93    |      61.86  |
| DocuNet-RoBERTa (large) | 62.23±0.12 | 64.12±0.14 | 62.39 | 64.55 |

### Document-level Relation Extraction on [CDR and GDA](https://github.com/fenchri/edge-oriented-graph)

| Model  |    CDR    | GDA |
| :----------------: | :----------------: | :----------------: |
| DocuNet-SciBERT (base) | 76.3±0.40    | 85.3±0.50  |




# Papers for the Project & How to Cite
If you use or extend our work, please cite the following paper:

```
@inproceedings{ijcai2021-551,
  title     = {Document-level Relation Extraction as Semantic Segmentation},
  author    = {Zhang, Ningyu and Chen, Xiang and Xie, Xin and Deng, Shumin and Tan, Chuanqi and Chen, Mosha and Huang, Fei and Si, Luo and Chen, Huajun},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {3999--4006},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/551},
  url       = {https://doi.org/10.24963/ijcai.2021/551},
}
```
