# DeepKE

DeepKE 是基于 Pytorch 的深度学习中文关系抽取处理套件。

## 环境依赖:

- python >= 3.6
- torch >= 1.0
- jieba >= 0.38
- matplotlib >= 3.0
- pytorch_transformers >= 1.2
- scikit_learn >= 0.20



## 主要目录

```
├── bert_pretrained         # 使用 bert 时存放的预训练模型参数
│ ├── vocab.txt             # BERT 模型词表
│ ├── config.json           # BERT 模型结构的配置文件
│ ├── pytorch_model.bin     # 预训练模型参数
├── checkpoints             # 保存训练后的模型参数
├── data                    # 数据目录
│ ├── origin                # 训练使用的原始数据集
│ │ ├── train.csv           # 训练数据集
│ │ ├── test.csv            # 测试数据集
│ │ ├── relation.txt        # 关系种类
├── deepke
│ ├── model                 # 模型目录
│ │ ├── BasicModule.py      # 模型基本配置
│ │ ├── Embedding.py        # Embeddding 模块
│ │ ├── CNN.py              # CNN & PCNN 模型
│ │ ├── RNN.py              # BiLSTM 模型
│ │ ├── GCN.py              # GCN 模型
│ │ ├── Transformer.py      # Transformer 模型
│ │ ├── Capsule.py          # Capsule 模型
│ │ ├── LM.py               # 语言预训练 模型
│ ├── config.py             # 配置文件
│ ├── vocab.py              # 词汇表构建函数
│ ├── preprocess.py         # 训练前预处理数据
│ ├── dataset.py            # 训练时批处理输入数据
│ ├── trainer.py            # 训练迭代函数
│ ├── utils.py              # 工具函数
├── main.py                 # 主入口文件
├── README.md               # read me 文件
```

## 快速开始

数据为 csv 文件，样式范例为：


sentence|relation|head|head_type|head_offset|tail|tail_type|tail_offset
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
谢万松，字树人，湖北省武汉市人，武汉钢铁集团公司联合焦化公司退体职工，生于1940年|出生地|谢万松|人物|0|湖北省武汉市|地点|8
《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧|导演|娘家的故事第二部|影视作品|1|张玲|人物|11
九玄珠是在纵横中文网连载的一部小说，作者是龙马|连载网站|九玄珠|网络小说|0|纵横中文网|网站|5
个人简介梁信强，男，2010年广州亚运会中国澳门代表团成员|国籍|梁信强|人物|4|中国|国家|20

- 安装依赖： `pip install -r requirements.txt`

- 存放数据：在 `data/origin` 文件夹下存放训练数据。训练文件主要有三个文件。

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `relation.txt`：存放关系种类

- 开始训练：python main.py

- 每次训练的结果会保存在 `checkpoints` 文件夹下，格式为：`{model_name}_{epoch}_{time}.pth`。

## 具体介绍

见 [wiki](https://github.com/zjunlp/deepke/wiki)


## 备注

使用语言预训练模型时，要提前下载好预训练好的参数，放到 `pretrained` 文件夹内。

另外数据量较小时，直接使用如12层的bert，效果并不理想，反而层数调低些收敛更快效果更好。如调整到3、6层数，具体看训练的数据量大小。

## 后续工作

- [ ] 添加经典实体关系联合抽取模型
- [ ] 添加 web 页面，以供预测输入句子信息可视化
