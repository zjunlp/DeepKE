## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/dev/example/triple/PRGC/README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

> python == 3.8

- torch == 1.10
- hydra-core == 1.3.0
- tensorboard == 2.4.1
- matplotlib == 3.4.1
- scikit-learn == 0.24.1
- transformers==4.20.0
- jieba == 0.42.1
- wandb == 0.13.9
- pandas == 1.5.3
- deepke 

### 克隆代码

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/triple/PRGC
```

### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/triple/PRGC/data.tar.gz```在此目录下

  - 数据集 [CMeIE](https://tianchi.aliyun.com/dataset/95414)/ [NYT](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view)/ [NYT*](https://github.com/weizhepei/CasRel/tree/master/data/NYT)/ [WebNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)/ [WebNLG*](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG) 被保存在`data`文件夹下:
    - `rel2id.json`：存放关系种类

    - `test_triples.json`： 存放测试数据集

    - `train_triples.json`: 存放训练数据集

    - `val_triples.json`：存放验证数据集
  
- 获得BERT预训练模型
  - 下载 [BERT-Base-Cased](https://huggingface.co/bert-base-cased)/ [BERT-Base-chinese](https://huggingface.co/bert-base-chinese) 放在 ./pretrain_models 文件夹下.
  - 重命名 config.json 为 bert_config.json
  - 把文件夹名中的 '-' 替换为 '_' 

- 训练

  - 训练的参数、模型路径和配置位于 ./conf 文件夹下，用户可以在训练前修改它们

    ```bash
    python run.py
    ```

  - 模型默认存储在 ./model 文件夹下

  - 训练的记录默认存储在 ./logs 文件夹下

- 预测

  ```bash
  python predict.py
  ```


