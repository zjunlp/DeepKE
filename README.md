# Deepke

deepke 是基于 Pytorch 的中文关系抽取处理套件。


## 环境依赖:

- python >= 3.6
- torch >=1.0.0
- numpy >= 1.14.2
- tqdm >= 4.28.1
- jieba >= 0.39


## 主要目录

```
├── checkpoints          # 保存训练后的模型参数
├── data                 # 数据目录
│ ├── origin             # 训练使用的原始数据集 
│   ├── train.csv        # 训练数据集
│   ├── valid.csv        # 测试数据集
│   ├── relation.txt     # 关系种类
├── models               # 模型目录
│ ├── __init__.py
│ ├── LSTM4VarLenSeq.py  # 定制LSTM用于可变长度的文本
│ ├── BiLSTM.py          # BiLSTM 模块
│ ├── Attention.py       # Attention 模块
│ ├── PCNN.py            # PCNN_ATT 模块
├── config.py            # 配置与超参
├── dataset.py           # 训练时批处理输入数据处理方法
├── main.py              # 程序主入口
├── metric.py            # 性能测试工具
├── README.md
├── utils.py             # 工具函数
├── vocab.py             # 构建词典
```

## 使用方法

数据为csv文件，样式范例为：

| sentence                   | entity1 | entity2 | offset1 | offset2 | relation |
|----------------------------|---------|---------|---------|---------|----------|
| 杨军接任铜陵有色董事长                | 杨军      | 铜陵有色    | 0       | 4       | 董事长      |
| 团贷网联合创始人张林获选鸿特精密董事长        | 张林      | 鸿特精密    | 8       | 12      | 董事长      |
| 恒信东方蹭海南热点配合减持公司及董事长孟宪民均遭处分 | 孟宪民     | 恒信东方    | 19      | 0       | 董事长      |


- 安装依赖： `pip install -r requirements.txt`

- 存放数据：在 `data/origin` 文件夹下存放训练数据。训练文件主要有三个文件。

    - `train.csv`：存放训练数据集

    - `valid.csv`：存放验证数据集

    - `relation.txt`：存放关系种类
   
- 训练：python main.py

- 每次训练的结果会保存在 `snapshot` 文件夹下，格式为：`{model_name}_{time}_{epoch}.pt`。


## 具体介绍

