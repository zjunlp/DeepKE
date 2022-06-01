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


<<<<<<< HEAD
=======
## 直接使用模型
我们使用[DUIE数据集](https://ai.baidu.com/broad/download?dataset=dureader)，并将其关系类型与实体类型与[cnSchema](https://github.com/OpenKG-ORG/cnSchema)对齐。cnSchema是面向中文信息处理，利用先进的知识图谱、自然语言处理和机器学习技术，融合结构化与文本数据，支持快速领域知识建模，支持跨数据源、跨领域、跨语言的开放数据自动化处理，为智能机器人、语义搜索、智能计算等新兴应用市场提供schema层面的支持与服务。

cnSchema基于的原则
* 完全开放，源自schema.org，OpenKG自主发布的Web Schema
* 立足中文，对接世界
* 面向应用，支持开放数据生态
* 社区共识，知识图谱专家指导

在这之上使用`chinese-bert-wwm`和`chinese-roberta-wwm-ext`为基础训练了DeepKE-cnschema(NER)模型。模型所使用的超参数为所给的参数。最终经过训练后可以得到如下表的效果

<table>
	<tr>
		<th>模型</th>
		<th>P</th>
		<th>R</th>
		<th>F1</th>
	</tr>
	<tr>
		<td>chinese-roberta-wwm-ext(micro)</td>
		<td>0.8028</td>
		<td>0.8612</td>
		<td>0.8310</td>
	</tr>
  <tr>
		<td>chinese-roberta-wwm-ext(macro)</td>
		<td>0.6990</td>
		<td>0.7295</td>
		<td>0.7021</td>
	</tr>
  <tr>
		<td>chinese-bert-wwm(micro)</td>
		<td>0.7841</td>
		<td>0.8587</td>
		<td>0.8197</td>
	</tr>
  <tr>
		<td>chinese-bert-wwm(macro)</td>
		<td>0.6921</td>
		<td>0.7393</td>
		<td>0.7078</td>
	</tr>
	
</table>

使用者可以直接下载[模型](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)进行预测，步骤如下：

1、只需将下载文件夹命名为`checkpoints`

2、只需修改 `predict.yaml`中的参数`text`为需要预测的文本

3、运行```python predict.py```即可直接进行预测使用。

如果需要使用其他模型进行训练，步骤如下：

1、也可先下载[数据集](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg)，将其放入命名为`data`的文件夹中

2、将`conf`文件夹中的`train.yaml`中的`bert_model`修改为指定模型

3、修改[源码](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/tools/preprocess.py)中的get_labels函数，返回的标签为所给`type.txt`中所用到的标签(这一步需要通过python setup.py install方式安装才能生效)

4、再运行```python run.py```即可进行训练。

使用训练好的模型，只需输入句子“《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽”，运行```python predict.py```后可得到结果，结果显示“星空黑夜传奇”实体类型为经过cnschema对齐后的“网络小说”，“起点中文网”为“网站”。

>>>>>>> a35345d5e74f92cb57dcf7288a7c4e8208de7fd3

### 模型内容

BERT
