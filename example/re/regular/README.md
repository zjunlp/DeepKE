## 快速上手

### 环境依赖

> python == 3.8

- torch == 1.5
- hydra-core == 1.0.6
- tensorboard == 2.4.1
- matplotlib == 3.4.1
- scikit-learn == 0.24.1
- transformers == 4.5.0
- jieba == 0.42.1
- deepke 

### 克隆代码
```
git clone git@github.com:zjunlp/DeepKE.git
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境,然

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据：在 `data/origin` 文件夹下存放训练数据。训练文件主要有三个文件。

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `test.csv`：存放测试数据集

  - `relation.csv`：存放关系种类

- 开始训练：```python run.py``` (训练所用到参数都在conf文件夹中，修改即可)

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

- 进行预测 ```python predict.py```


## 模型内容
1、CNN

2、RNN

3、Capsule

4、GCN

5、Transformer

6、预训练模型
