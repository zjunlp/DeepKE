## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

> python == 3.8 

- pytorch-transformers == 1.2.0
- torch == 1.5.0
- hydra-core == 1.0.6
- seqeval == 1.2.2
- tqdm == 4.60.0
- matplotlib == 3.4.1
- deepke



### 克隆代码

```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/standard
```



### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖：`pip install -r requirements.txt`



### 使用数据进行训练预测

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/ner/standard/data.tar.gz```在此目录下

  在`data`文件夹下存放数据：
  
  - `train.txt`：存放训练数据集
  - `valid.txt`：存放验证数据集
  - `test.txt`：存放测试数据集
- 开始训练：```python run.py``` (训练所用到参数都在conf文件夹中，修改即可)

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

- 进行预测 ```python predict.py```



### 模型内容

BERT
