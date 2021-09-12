# 快速上手

## 环境依赖

> python >= 3.7 

- pytorch-transformers==1.2.0
- torch==1.2.0
- seqeval==0.0.5
- tqdm==4.31.1
- nltk==3.4.5



## 克隆代码

```
git clone git@github.com:zjunlp/DeepKE.git
```



## 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖：`pip install -r requirements.txt`



## 使用数据进行训练预测

- 存放数据：在`data`文件夹下存放数据集。主要有三个文件：
  - `train.txt`：存放训练数据集
  - `valid.txt`：存放验证数据集
  - `test.txt`：存放测试数据集

- 先进行训练，训练后的模型参数保存在out_ner文件夹中

  ```
  python run.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_ner --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1
  ```

- 再进行预测

  - 执行以下命令运行示例`python predict.py`

  - 如果需要指定NER的文本，可以利用--text参数指定，如：

    ````
    python predict.py --text="It was one o'clock when we left Lauriston Gardens and Sherlock Holmes led me to Metropolitan Police Service.."
    ````



## 模型内容

BERT
