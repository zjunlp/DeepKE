# 快速上手

## 克隆代码

```
git clone git@github.com:xxupiano/BERTNER.git
```



## 配置环境

创建python虚拟环境（python>=3.7）

安装依赖库

```
pip install -r requirements.txt
```



## 使用工具

 先进行训练

```
python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_ner --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1
```

再进行预测

- 修改main.py中text为需要进行NER的句子

- ```
  python main.py
  ```

