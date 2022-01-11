## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ae/standard/README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

> python == 3.8

- torch == 1.5
- hydra-core == 1.0.6
- tensorboard == 2.4.1
- matplotlib == 3.4.1
- scikit-learn == 0.24.1
- transformers == 3.4.0
- jieba == 0.42.1
- deepke 

### 克隆代码
```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ae/standard
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据：  可先下载数据 ```wget 120.27.214.45/Data/ae/standard/data.tar.gz```至此目录下

  解压后`data/origin` 文件夹下存放来训练数据：

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `test.csv`：存放测试数据集

  - `attribute.csv`：存放属性种类

- 开始训练：```python run.py``` (训练所用到参数都在conf文件夹中，修改即可；使用LM时，可修改'lm_file'使用下载至本地的模型)

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

- 进行预测 ```python predict.py```


## 模型内容
1、CNN

2、RNN

3、Capsule

4、GCN

5、Transformer

6、预训练模型
