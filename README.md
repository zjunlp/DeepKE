

## 介绍

基于 Pytorch 的中文关系抽取。 


## 环境:

- Python 3.6
- Pytorch 1.1.0
- jieba 0.39

## 主要目录

```
├── checkpoints          # 保存训练后的模型参数
├── data                 # 数据目录
│ ├── origin             # 训练使用的原始数据集 
│   ├── train.txt        # 训练数据集
│   ├── test.txt         # 测试数据集
│   ├── label.txt        # 关系种类
├── models               # 模型目录
│ ├── __init__.py
│ ├── LSTM4VarLenSeq.py  # 定制LSTM用于可变长度的文本
│ ├── BiLSTM_ATT.py      # BiLSTM_ATT 模型
│ ├── PCNN_ATT.py        # PCNN_ATT 模型
├── config.py            # 配置与超参
├── dataset.py           # 训练时批处理输入数据处理方法
├── main.py              # 程序主入口
├── metric.py            # 性能测试工具
├── README.md
├── utils.py             # 工具函数
├── vocab.py             # 构建词典
```

## 使用方法

数据样式为：
```
{"text": "...句子...", "entity1": "实体1", "entity2": "实体2", "relation": "关系"}
{"text": "...句子...", "entity1": "实体1", "entity2": "实体2", "relation": "关系"}
{"text": "...句子...", "entity1": "实体1", "entity2": "实体2", "relation": "关系"}
```


- 安装依赖： `pip install -r requirements.txt`

- 存放数据：在 `data/origin` 文件夹下存放训练数据。训练文件主要有三个文件。

    - `train.txt`：存放训练数据集

    - `test.txt`：存放测试数据集

    - `label.txt`：存放关系种类
   
- 训练：python main.py

- 每次训练的结果会保存在 `checkpoint` 文件夹下，格式为：`{model_name}_{time}_{epoch}.pkl`。
