# 快速上手

## 克隆代码

```
git clone git@github.com:zjunlp/DeepKE.git
```



## 配置环境

创建python虚拟环境（python>=3.7）

安装依赖库

```
pip install -r requirements.txt
```



## 使用工具

先进行训练，训练后的模型参数保存在out_ner文件夹中

```
python run.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_ner --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1
```

再进行预测<br>

执行以下命令运行示例

```
python predict.py
```
如果需要指定NER的文本，可以利用--text参数指定
```
python predict.py --text="Irene, a master student in Zhejiang University, Hangzhou, is traveling in Warsaw for Chopin Music Festival."
```
