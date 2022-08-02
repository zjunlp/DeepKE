## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/multimodal/README.md">English</a> | 简体中文 </b>
</p>

### 模型简介

**IFAformer**是一个具有隐式特征对齐的双模态Transformer模型，它在视觉和文本编码器上使用统一的Transformer架构，且无需显示设计模态对齐模块。

<div align=center>
<img src="mner_model.png" width="75%"height="75%"/>
</div>

### 实验结果

IFAformer的主要实验结果如下表所示：

<table>
	<tr>
		<th></th>
		<th>Methods</th>
		<th>Precision</th>
		<th>Recall</th>
		<th>F1</th>
	</tr>
	<tr>
		<td rowspan="3">text</td>
		<td>CNN-BiLSTM-(CRF)</td>
		<td>80.00</td>
		<td>78.76</td>
		<td>79.37</td>
	</tr>
	<tr>
		<td>BERT-(CRF)</td>
		<td>83.32</td>
		<td>83.57</td>
		<td>83.44</td>
	</tr>
	<tr>
		<td>MTB</td>
		<td>83.88</td>
		<td>83.22</td>
		<td>83.55</td>
	</tr>
	<tr>
		<td rowspan="5">text+image</td>
		<td>AdapCoAtt-BERT-(CRF)</td>
		<td>85.13</td>
		<td>83.20</td>
		<td>84.10</td>
	</tr>
	<tr>
		<td>VisualBERT_base</td>
		<td>84.06</td>
		<td>85.39</td>
		<td>84.72</td>
	</tr>
	<tr>
		<td>ViLBERT_base</td>
		<td>84.62</td>
		<td>85.47</td>
		<td>85.04</td>
	</tr>
	<tr>
		<td>UMT</td>
		<td>85.28</td>
		<td>85.34</td>
		<td>85.31</td>
	</tr>
	<tr>
		<td><b>IFAformer</b></td>
		<td><b>86.88</b></td>
		<td><b>87.91</b></td>
		<td><b>87.39</b></td>
	</tr>
</table>

### 环境依赖

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

<font color='red'> 注意! </font> 这里`transformers==3.4.0`是整个`DeepKE`的环境要求。但是在使用多模态部分时，`openai/clip-vit-base-patch32`预训练模型的加载需要`transformers==4.11.3`。因此推荐大家在huggingface上下载好[模型](https://huggingface.co/openai/clip-vit-base-patch32)后，采用本地路径的方法导入。

### 克隆代码

```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/multimodal
```

### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ``pip install -r requirements.txt``

### 使用数据进行训练预测

- 模型采用的数据是Twitter15和Twitter17，文本数据采用conll格式，更多信息可参考[UMT](https://github.com/jefferyYu/UMT/)
- 存放数据： 可先下载数据 ``wget 120.27.214.45/Data/ner/multimodal/data.tar.gz``在此目录下
- Twitter15包含以下数据：
  
  - `twitter15_detect`：使用RCNN检测子图
  - `twitter2015_aux_images`：使用visual grouding检测的子图
  - `twitter2015_images`： 原始图片
  - `train.txt`: 训练文本数据
  - `...`
- 开始训练：模型加载和保存位置以及配置可以在conf的 `.yaml`文件中修改
- 下载[PLM](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)并修改`train.yaml`和`predict.yaml`中的`vit_name'为PLM的绝对路径
  
  - `python run.py`
  - 训练好的模型默认保存在 `checkpoint`中，可以通过修改 `train.yaml`中的"save_path"更改保存路径
- 从上次训练的模型开始训练：设置 `.yaml`中的save_path为上次保存模型的路径
- 每次训练的日志保存路径默认保存在当前目录，可以通过 `.yaml`中的log_dir来配置
- 进行预测： 修改 `predict.yaml`中的load_path来加载训练好的模型。此外，我们提供了在Twitter2017数据集上训练好的[模型](https://drive.google.com/drive/folders/1ZGbX9IiNU3cLZtt4U8oc45zt0BHyElAQ?usp=sharing)供大家直接预测使用
- `python predict.py `

