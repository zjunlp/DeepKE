## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/few-shot/README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

### 克隆代码
```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/few-shot
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/re/few_shot/data.tar.gz```在此目录下

  在 `data` 文件夹下存放训练数据。模型采用的数据集是[SEMEVAL](https://semeval2.fbk.eu/semeval2.php?location=tasks#T11)，SEMEVAL数据集来自于2010年的国际语义评测大会中Task 8："Multi-Way Classification of Semantic Relations Between Pairs of Nominals"。

- SEMEVAL包含以下数据：

  - `rel2id.json`：关系标签到ID的映射

  - `temp.txt`：关系标签处理

  - `test.txt`： 测试集

  - `train.txt`：训练集

  - `val.txt`：验证集

- 开始训练：模型加载和保存位置以及配置可以在conf的`.yaml`文件中修改
  
  - 对数据集SEMEVAL进行few-shot训练：`python run.py` 

  - 训练好的模型默认保存在当前目录

- 从上次训练的模型开始训练：设置`.yaml`中的train_from_saved_model为上次保存模型的路径

- 每次训练的日志保存路径默认保存在当前目录，可以通过`.yaml`中的log_dir来配置

- 进行预测： `python predict.py `


## 模型内容
[KnowPrompt](https://arxiv.org/abs/2104.07650)
