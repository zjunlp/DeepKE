## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/standard/README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

> python == 3.8

- torch == 1.5
- hydra-core == 1.0.6
- tensorboard == 2.4.1
- matplotlib == 3.4.1
- scikit-learn == 0.24.1
- transformers == 3.4.0
- jieba == 0.42.1
- deepke 

### 克隆代码
```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/standard
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/re/standard/data.tar.gz```在此目录下

  在 `data/origin` 文件夹下存放训练数据：

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `test.csv`：存放测试数据集

  - `relation.csv`：存放关系种类

- 开始训练：```python run.py``` (训练所用到参数都在conf文件夹中，修改即可使用lm时，可修改'lm_file'使用下载至本地的模型)

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。

- 进行预测 ```python predict.py```

## 直接使用模型

我们使用[DUIE数据集](https://ai.baidu.com/broad/download?dataset=dureader)，并将其关系类型与实体类型与[cnSchema](https://github.com/OpenKG-ORG/cnSchema)对齐。cnSchema是面向中文信息处理，利用先进的知识图谱、自然语言处理和机器学习技术，融合结构化与文本数据，支持快速领域知识建模，支持跨数据源、跨领域、跨语言的开放数据自动化处理，为智能机器人、语义搜索、智能计算等新兴应用市场提供schema层面的支持与服务。

cnSchema基于的原则
* 完全开放，源自schema.org，OpenKG自主发布的Web Schema
* 立足中文，对接世界
* 面向应用，支持开放数据生态
* 社区共识，知识图谱专家指导

在这之上使用`chinese-bert-wwm`和`chinese-roberta-wwm-ext`为基础训练了DeepKE-cnschema(RE)模型。模型所使用的超参数为所给的参数。最终经过训练后可以得到如下表的效果

<table>
	<tr>
		<th>模型</th>
		<th>P</th>
		<th>R</th>
		<th>F1</th>
	</tr>
  <tr>
		<td>chinese-roberta-wwm-ext(macro)</td>
		<td>0.8761</td>
		<td>0.8598</td>
		<td>0.8665</td>
	</tr>
  <tr>
		<td>chinese-bert-wwm(macro)</td>
		<td>0.8742</td>
		<td>0.8582</td>
		<td>0.8639</td>
	</tr>
	
</table>


使用者可以直接下载[模型](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)使用,步骤如下：

1、修改 `predict.yaml`中的参数`fp`为下载文件的路径，`embedding.yaml`中`num_relations`为51

2、运行```python predict.py```即可直接进行预测使用。需要预测的文本及实体对通过终端返回给程序。

如果需要使用其他模型进行训练，步骤如下：

1、也可先下载[数据集](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)，将其重命名为`data`

2、将`conf`文件夹中的`train.yaml`为`lm`,`lm.yaml`中的`lm_file`修改为指定预训练模型，`embedding.yaml`中`num_relations`为51

3、运行```python run.py```即可进行训练。


使用训练好的模型，运行```python predict.py```后，只需输入的句子为“东眼山森林游乐区位于桃园县与台北县交界的山林间，因山形酷似“向东眺望的大眼睛”而得名，海拔高度在650~1212公尺之间，面积916公顷，是台湾北部面积最大的森林游乐区”，给定的实体对为“东眼山”和“1212公尺”，可得到结果，最终抽取出的关系为经过cnschema对齐后的“海拔”。




## 模型内容
1、CNN

2、RNN

3、Capsule

4、GCN

5、Transformer

6、预训练模型
