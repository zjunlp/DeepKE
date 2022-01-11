## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/document/README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

> python == 3.8

- torch == 1.5.0
- transformers == 3.4.0
- opt-einsum == 3.3.0
- ujson
- deepke

### 克隆代码
```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/d
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/re/document/data.tar.gz```在此目录下

  在 `data` 文件夹下存放训练数据。模型采用的数据集是[DocRED](https://github.com/thunlp/DocRED/tree/master/)，DocRED数据集来自于2010年的国际语义评测大会中Task 8："Multi-Way Classification of Semantic Relations Between Pairs of Nominals"。


- DocRED包含以下数据：

  - `dev.json`：验证集

  - `rel_info.json`：关系集

  - `rel2id.json`：关系标签到ID的映射

  - `test.json`：测试集

  - `train_annotated.json`：人工标注的训练集

  - `train_distant.json`：远程监督产生的训练集

- 开始训练：模型加载和保存位置以及配置可以在conf的`.yaml`文件中修改
  
  - 在数据集DocRED中训练：`python run.py` 

  - 训练好的模型保存在当前目录下

- 从上次训练的模型开始训练：设置`.yaml`中的train_from_saved_model为上次保存模型的路径

- 每次训练的日志保存路径默认保存在根目录，可以通过`.yaml`中的log_dir来配置

- 进行预测： `python predict.py`

  - 预测生成的`result.json`保存在根目录


## 模型内容
[DocuNet](https://arxiv.org/abs/2106.03618)
