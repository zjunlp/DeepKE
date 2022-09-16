## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/multimodal/README.md">English</a> | 简体中文 </b>
</p>

### 模型简介

**IFAformer**是一个具有隐式特征对齐的双模态Transformer模型，它在视觉和文本编码器上使用统一的Transformer架构，且无需显示设计模态对齐模块。

<div align=center>
<img src="mre_model.png" width="75%" height="75%"/>

</div>

### 实验结果

IFAformer的主要实验结果如下表所示：

<table>
	<tr>
		<th></th>
		<th>Methods</th>
		<th>Acc</th>
		<th>Precision</th>
		<th>Recall</th>
		<th>F1</th>
	</tr>
	<tr>
		<td rowspan="3">text</td>
		<td>PCNN*</td>
		<td>73.36</td>
		<td>69.14</td>
		<td>43.75</td>
		<td>53.59</td>
	</tr>
	<tr>
		<td>BERT*</td>
		<td>71.13</td>
		<td>58.51</td>
		<td>60.16</td>
		<td>59.32</td>
	</tr>
	<tr>
		<td>MTB*</td>
		<td>75.34</td>
		<td>63.28</td>
		<td>65.16</td>
		<td>64.20</td>
	</tr>
	<tr>
		<td rowspan="4">text+image</td>
	</tr>
	<tr>
		<td>BERT+SG+Att</td>
		<td>74.59</td>
		<td>60.97</td>
		<td>66.56</td>
		<td>63.64</td>
	</tr>
	<tr>
		<td>ViLBERT</td>
		<td>74.89</td>
		<td>64.50</td>
		<td>61.86</td>
		<td>63.61</td>
	</tr>
	<tr>
		<td>MEGA</td>
		<td>76.15</td>
		<td>64.51</td>
		<td>68.44</td>
		<td>66.41</td>
	</tr>
	<tr>
		<td rowspan="4">Ours</td>
	</tr>
	<tr>
		<td>Vanilla IFAformer</td>
		<td>87.75</td>
		<td>69.90</td>
		<td>68.11</td>
		<td>68.99</td>
	</tr>
	<tr>
		<td>&emsp;w/o Text Attn.</td>
		<td>76.21</td>
		<td>66.95</td>
		<td>61.72</td>
		<td>64.23</td>
	</tr>
	<tr>
		<td>&emsp;w/ Visual Objects</td>
		<td><b>92.38</b></td>
		<td><b>82.59</b></td>
		<td><b>80.78</b></td>
		<td><b>81.67</b></td>
	</tr>
</table>

### 环境依赖

- python == 3.8
- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

<font color='red'> 注意! </font> 这里`transformers==3.4.0`是整个`DeepKE`的环境要求。但是在使用多模态部分时，`openai/clip-vit-base-patch32`预训练模型的加载需要`transformers==4.11.3`。因此推荐大家在huggingface上下载好[模型](https://huggingface.co/openai/clip-vit-base-patch32)后，采用本地路径的方法导入。

### 克隆代码

```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/multimodal
```

### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ``pip install -r requirements.txt``

### 使用数据进行训练预测

- 模型采用的数据是[MNRE](https://github.com/thecharm/Mega)，MRE数据集来自于[Multimodal Relation Extraction with Efficient Graph Alignment](https://dl.acm.org/doi/10.1145/3474085.3476968)
- 存放数据： 可先下载数据 ``wget 120.27.214.45/Data/re/multimodal/data.tar.gz``在此目录下
- MNRE包含以下数据：
  
  - `img_detect`：使用RCNN检测的实体
  - `img_vg`：使用visual grounding检测的实体
  - `img_org`： 原图像
  - `txt`: 文本你数据
  - `vg_data`：绑定原图和 `img_vg`
  - `ours_rel2id.json` 关系集合
- 我们使用RCNN检测的实体和visual grounding检测的实体作为视觉局部信息，其中，RCNN检测可以通过[faster_rcnn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)完成，visual grounding可以通过[onestage_grounding
  ](https://github.com/zyang-ur/onestage_grounding)完成。
- 开始训练：模型加载和保存位置以及配置可以在conf的 `.yaml`文件中修改
  
  - `python run.py`
  - 训练好的模型默认保存在 `checkpoint`中，可以通过修改 `train.yaml`中的"save_path"更改保存路径
- 从上次训练的模型开始训练：设置 `.yaml`中的save_path为上次保存模型的路径
- 每次训练的日志保存路径默认保存在当前目录，可以通过 `.yaml`中的log_dir来配置
- 进行预测： 修改 `predict.yaml`中的load_path来加载训练好的模型。此外，我们提供了在MNRE数据集上训练好的[模型](https://drive.google.com/drive/folders/11T0t1NHSMq5GzORBKv2Rjm2Bbq_RNLrc?usp=sharing)供大家直接预测使用。
- `python predict.py `

