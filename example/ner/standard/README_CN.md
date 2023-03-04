## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/README.md">English</a> | 简体中文 </b>
</p>

### 模型内容

本项目实现了Standard场景下NER任务的提取模型，对应路径分别为：
* [BiLSTM-CRF](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/models/BiLSTM_CRF.py)
* [Bert](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/models/InferBert.py)
* [W2NER](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/w2ner)


### 实验结果
| 模型        | 准确率   | 召回率   | f1值   | 推理速度([People's Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily)) |
|-----------|-------|-------|-------|------------------------------------------------------------------------------------------------------|
| BERT      | 91.15 | 93.68 | 92.40 | 106s                                                                                                 |
| BiLSTM-CRF | 92.11 | 88.56 | 90.29 | 39s                                                                                                  |
| W2NER     | 96.76 | 96.11 | 96.43 | -                                                                                                    |
### 环境依赖

> python == 3
```bash
pip install -r requirements.txt
```



### 克隆代码

```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/standard
```



### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖：`pip install -r requirements.txt`

### 参数设置

#### 1.model parameters

[`conf/hydra/model/*.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf/hydra/model)路径下为模型的参数配置，例如控制模型的隐藏层维度、是否Case Sensitive......

#### 2.other parameters

环境路径以及训练过程中的其他超参数在[`train.yaml`、`custom.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf)中进行设置。

> 注： 训练过程中所用到的词典
> 
> 使用`Bert`模型时vocab来自huggingface的预训练权重
> 
> 使用`BiLSTM_CRF`则需要根据训练集构建词典，并存储在pkl文件中供预测和评价使用。(配置为`lstmcrf.yaml`中的`model_vocab_path`属性)

### 使用数据进行训练预测

- 支持三种类型文件格式，包含json格式、docx格式以及txt格式，详细可参考`data`文件夹。模型采用的数据是People's Daily(中文NER)，文本数据采用{word, label}对格式
- **默认支持中文数据集，如需使用英文数据集，prediction前需修改config.yaml中的lan，并安装nltk，下载nltk.download('punkt')**

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/ner/standard/data.tar.gz```在此目录下

  在`data`文件夹下存放数据：
  
  - `train.txt`：存放训练数据集
  - `valid.txt`：存放验证数据集
  - `test.txt`：存放测试数据集
- 开始训练(可根据**目标场景**选Bert或者BiLSTM-CRF或者W2NER)：

  1. ```python run_bert.py``` (修改[config.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/conf/config.yaml)中hydra/model为bert，bert超参设置在[bert.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/conf/hydra/model/bert.yaml)中，训练所用到参数都在conf文件夹中，修改即可).该任务支持多卡训练，修改trian.yaml中的use_multi_gpu参数为True，run_bert.py中osviron['CUDA_VISIBLE_DEVICES']为指定gpu，以逗号为间隔，第一张卡为计算主卡，需使用略多内存。
  2. ```python run_lstmcrf.py``` (BiLSTM-CRF超参设置在`lstmcrf.yaml`中，训练所用到参数都在conf文件夹中，修改即可)
  3. ```cd w2ner ---> python run.py``` (w2nerF超参设置在`model.yaml`中，训练所用到参数都在conf文件夹中，修改即可。其中`device`为指定GPU的编号，若只有单卡GPU，设置为0)

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

- 模型加载和保存位置以及配置可以在conf的 `*.yaml`文件中修改

- 进行预测 ```python predict.py```

### 样本自动化打标

如果您只有文本数据和对应的词典，而没有规范的训练数据。

您可以通过自动化打标方法得到弱监督的格式化训练数据，请确保：

- 提供高质量的词典
- 充足的文本数据

<p align="left">
<a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data/README.md">prepare-data</a> </b>
</p>
