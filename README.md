# DeepKE

DeepKE 是基于 Pytorch 的深度学习中文关系抽取处理套件。

## 环境依赖:

> python >= 3.6

- torch >= 1.2
- hydra-core >= 0.11
- tensorboard >= 2.0
- matplotlib >= 3.1
- transformers >= 2.0
- jieba >= 0.39
- ~~pyhanlp >= 0.1.57~~（中文句法分析使用，但是在多句时效果也不好。。求推荐有比较好的中文句法分析）


## 主要目录

```
├── conf                      # 配置文件夹
│  ├── config.yaml            # 配置文件主入口
│  ├── preprocess.yaml        # 数据预处理配置
│  ├── train.yaml             # 训练过程参数配置
│  ├── hydra                  # log 日志输出目录配置
│  ├── embedding.yaml         # embeding 层配置
│  ├── model                  # 模型配置文件夹
│  │  ├── cnn.yaml            # cnn 模型参数配置
│  │  ├── rnn.yaml            # rnn 模型参数配置
│  │  ├── capsule.yaml        # capsule 模型参数配置
│  │  ├── transformer.yaml    # transformer 模型参数配置
│  │  ├── gcn.yaml            # gcn 模型参数配置
│  │  ├── lm.yaml             # lm 模型参数配置
├── pretrained                # 使用如 bert 等语言预训练模型时存放的参数
│  ├── vocab.txt              # BERT 模型词表
│  ├── config.json            # BERT 模型结构的配置文件
│  ├── pytorch_model.bin      # 预训练模型参数
├── data                      # 数据目录
│  ├── origin                 # 训练使用的原始数据集
│  │  ├── train.csv           # 训练数据集
│  │  ├── valid.csv           # 验证数据集
│  │  ├── test.csv            # 测试数据集
│  │  ├── relation.csv        # 关系种类
│  ├── out                    # 预处理数据后的存放目录
├── module                    # 可复用模块
│  ├── Embedding.py           # embedding 层
│  ├── CNN.py                 # cnn
│  ├── RNN.py                 # rnn
│  ├── Attention.py           # attention
│  ├── Transformer.py         # transformer
│  ├── Capsule.py             # capsule
│  ├── GCN.py                 # gcn
├── models                    # 模型目录
│  ├── BasicModule.py         # 模型基本配置
│  ├── PCNN.py                # PCNN / CNN 模型
│  ├── BiLSTM.py              # BiLSTM 模型
│  ├── Transformer.py         # Transformer 模型
│  ├── LM.py                  # Language Model 模型
│  ├── Capsule.py             # Capsule 模型
│  ├── GCN.py                 # GCN 模型
├── test                      # pytest 测试目录
├── tutorial-notebooks        # simple jupyter notebook tutorial
├── utils                     # 常用工具函数目录
├── metrics.py                # 评测指标文件
├── serializer.py             # 预处理数据过程序列化字符串文件
├── preprocess.py             # 训练前预处理数据文件
├── vocab.py                  # token 词表构建函数文件
├── dataset.py                # 训练过程中批处理数据文件
├── trainer.py                # 训练验证迭代函数文件
├── main.py                   # 主入口文件（训练）
├── predict.py                # 测试入口文件（测试）            
├── README.md                 # read me 文件
```

## 快速开始

数据为 csv 文件，样式范例为：


sentence|relation|head|head_offset|tail|tail_offset
:---:|:---:|:---:|:---:|:---:|:---:
《岳父也是爹》是王军执导的电视剧，由马恩然、范明主演。|导演|岳父也是爹|1|王军|8
《九玄珠》是在纵横中文网连载的一部小说，作者是龙马。|连载网站|九玄珠|1|纵横中文网|7
提起杭州的美景，西湖总是第一个映入脑海的词语。|所在城市|西湖|8|杭州|2

- 安装依赖： `pip install -r requirements.txt`

- 存放数据：在 `data/origin` 文件夹下存放训练数据。训练文件主要有三个文件。更多数据建议使用百度数据库中[Knowledge Extraction](http://ai.baidu.com/broad/download)。

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `test.csv`：存放测试数据集

  - `relation.csv`：存放关系种类

- 开始训练：python main.py

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

## 具体介绍

见 [wiki](https://github.com/zjunlp/deepke/wiki)


## 备注（常见问题）

1. 使用 Anaconda 时，建议添加国内镜像，下载速度更快。如[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

1. 使用 pip 时，建议使用国内镜像，下载速度更快，如阿里云镜像。

1. 安装后提示 `ModuleNotFoundError: No module named 'past'`，输入命令 `pip install future` 即可解决。

1. 使用 `python main.py --help` 可以查看所有可配置参数，并定制修改参数结果。参数为 bool 值的，可以用 `1，0` 代替 `True, False`。

    - 如 `python main.py epoch=100 batch_size=128 use_gpu=False`

1. 使用 `python main.py xxx=xx,xx  -m` 可以多任务处理程序。

    - 如 `python main.py model=cnn,rnn,lm  chinese_split=0,1  -m` 可以生成 3*2=6 个子任务。

1. 中文英文在数据预处理上有很多不同之处，`serializer.py` 用来专门序列化句子为 tokens。中文分词使用的是 jieba 分词。

    - 英文序列化要求：大小写、特殊标点字符处理、特殊英文字符是否分词、是否做 word-piece 处理等。
    
    - 中文序列化要求：是否分词、遇到英文字母是否大小写处理、是否将英文单词拆分按照单独字母处理等。
    
1. PCNN 预处理时，需要按照 head tail 的位置，将句子分为三段，做 piece wise max pooling。如果句子本身无法分为三段，就无法用统一的预处理方式处理句子。
    
    - 比如句子为：`杭州西湖`，不管怎么分隔都不能分隔为三段。
    
    - 原文分隔三段的方式为：`[...head,  ...,  tail....]`，当然也可以分隔为：`[...,  head...tail,  ....]`，或者 `[...head,  ...tail,  ....]`  或者 `[...,  head...,  tail...]` 等。具体效果没多少区别。
    
1. PCNN 为什么不比 CNN 好，甚至更差？？

    - 本人在跑百度的数据集，也发现 PCNN 效果并没有想象中的比 CNN 有提升，甚至大多时候都不如 CNN 那种直接 max pooling的结果。百度的 [ARNOR](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2019-ARNOR) 结果也是 PCNN 并不一定比 CNN 好。

1. 使用语言预训练模型时，在线安装下载模型比较慢，更建议提前下载好，存放到 `pretrained` 文件夹内。具体存放文件要求见文件夹内的 `readme.md`。

1. 数据量较小时，直接使用如12层的 BERT，效果并不理想。此时可采取一些处理方式改善效果：
    
    - 数据量较小时层数调低些，如设置为2、3层。
    
    - 按照 BERT 训练方式，对新任务语料按照语言模型方式预训练。
    
1. 在单句上使用 GCN 时，需要先做句法分析，构建出词语之间的邻接矩阵（句法树相邻的边值设为1，不相邻为0）。
    
    - ~~目前使用的是 `pyhanlp` 工具构建语法树。这个工具需要按照 java 包，具体使用见 [pyhanlp](https://github.com/hankcs/pyhanlp) 的介绍。~~ pyhanlp 在多句时效果也不理想，很多时候把整个单句当作一个节点。


## 后续工作

- [x] 重构代码，将模型可复用部分单独提取出来
- [ ] 添加经典实体关系联合抽取模型
- [ ] 添加 web 页面，以供预测输入句子信息可视化


> Author: [余海阳](mailto:yuhaiyang@zju.edu.cn)

> Organization: [浙江大学知识引擎实验室](http://openkg.cn/)

