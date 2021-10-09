## 快速上手

### 环境依赖

> python == 3.8

- numpy == 1.20.3
- tokenizers == 0.10.3
- torch == 1.8.0
- regex == 2021.4.4
- transformers == 4.7.0
- tqdm == 4.49.0
- activations == 0.1.0
- dataclasses == 0.6
- file_utils == 0.0.1
- flax == 0.3.4
- utils == 1.0.1
- deepke 

### 克隆代码
```
git clone git@github.com:zjunlp/DeepKE.git
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据：在 `data` 文件夹下存放训练数据。模型采用的数据集是[DocRED](https://github.com/thunlp/DocRED/tree/master/)，DocRED数据集来自于2010年的国际语义评测大会中Task 8："Multi-Way Classification of Semantic Relations Between Pairs of Nominals"。

- DocRED包含以下数据：

  - `dev.json`：验证集

  - `rel_info.json`：关系集

  - `rel2id.json`：关系标签到ID的映射

  - `test.json`：测试集

  - `train_annotated.json`：训练集

- 开始训练：模型加载和保存位置以及配置可以在conf的`.yaml`文件中修改
  
  - 在数据集DocRED中训练：`python run.py` 

- 每次训练的日志保存路径可以通过`.yaml`中的log_dir来配置。

- 进行预测： `python predict.py `


## 模型内容
DocuNet
