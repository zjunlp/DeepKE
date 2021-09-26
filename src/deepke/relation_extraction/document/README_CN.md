[**中文**](https://github.com/zjunlp/DocRE/blob/master/README_CN.md) | [**English**](https://github.com/zjunlp/DocRE/blob/master/README.md)

<p align="center">
  	<font size=7><strong>DocuNet：一个基于语义分割方法实现文档级关系抽取的模型</strong></font>
</p>

这是针对我们[**DocuNet**](https://github.com/zjunlp/DocuNet)项目的官方实现代码。这个模型是在**[Document-level Relation Extraction as Semantic Segmentation](https://www.ijcai.org/proceedings/2021/551)**论文中提出来的，该论文已被**IJCAI2021**主会录用。


# 项目成员
陈想，谢辛，邓淑敏，张宁豫，陈华钧。


# 项目简介
本文创新性地提出DocuNet模型，首次将文档级关系抽取任务类比于计算机视觉中的语义分割任务。


# 环境要求
需要按以下命令去配置项目运行环境：

```运行准备
pip install -r requirements.txt
```
# 模型训练

## DocRED


请运行以下命令在DocRED中训练DocuNet模型：

```bash
>> bash scripts/run_docred.sh # use BERT/RoBERTa by setting --transformer-type
```
## CDR和GDA

请运行以下命令在CDR和GDA中训练DocuNet模型：

```bash
>> bash scripts/run_cdr.sh  # for CDR
>> bash scripts/run_gda.sh  # for GDA
```

数据集GDR和CDA可以根据[edge-oriented graph](https://github.com/fenchri/edge-oriented-graph)指南获取。


# 评估效果
>要评估论文中的训练模型，您可以在训练脚本中设置 `--load_path` 参数。程序会自动记录评估结果。对于 DocRED，它将生成一个官方评估格式的测试文件 `result.json`。您可以压缩并提交给 Colab 以获得官方测试分数。


# 结果

我们的模型达到了以下的性能：

### 在[DocRED](https://github.com/thunlp/DocRED)上的文档级关系抽取

| 模型     | Ign F1 on Dev | F1 on Dev | Ign F1 on Test | F1 on Test |
| :----------------: |:--------------: | :------------: | ------------------ | ------------------ |
| DocuNet-BERT (base) |  59.86±0.13 |   61.83±0.19 |     59.93    |      61.86  |
| DocuNet-RoBERTa (large) | 62.23±0.12 | 64.12±0.14 | 62.39 | 64.55 |


### 在[CDR和GDA](https://github.com/fenchri/edge-oriented-graph)上的文档级关系抽取

| 模型  |    CDR    | GDA |
| :----------------: | :----------------: | :----------------: |
| DocuNet-SciBERT (base) | 76.3±0.40    | 85.3±0.50  |


# 有关论文
如果您使用或拓展我们的工作，请引用以下论文：

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



