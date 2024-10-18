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

  支持三种类型文件格式，包含json格式、xlsx格式以及csv格式，详细可参考`data`文件夹。在 `data/origin` 文件夹下存放训练数据：

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `test.csv`：存放测试数据集

  - `relation.csv`：存放关系种类

- 开始训练：```python run.py``` (训练所用到参数都在conf文件夹中，修改即可使用lm时，可修改'lm_file'使用下载至本地的模型),该任务支持多卡训练，修改`trian.yaml`中的`use_multi_gpu`参数为True，`gpu_ids`设置为所选gpu，以逗号为间隔，第一张卡为计算主卡，需使用略多内存。设置`show_plot`为True可对当前epoch的loss进行可视化展示，默认为False。

- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存在 `checkpoints` 文件夹内。
- 修改 [predict.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/re/standard/conf/predict.yaml)中的fp为用于预测的模型或checkpoint路径，需使用模型的绝对路径。如`xxx/checkpoints/2019-12-03_17-35-30/cnn_epoch21.pth`。
- 进行预测 ```python predict.py```

## 模型内容
1、CNN

2、RNN

3、Capsule

4、GCN （基于论文["Graph Convolution over Pruned Dependency Trees Improves Relation Extraction"](https://aclanthology.org/D18-1244.pdf))

5、Transformer

6、预训练模型

## 数据标注

如果您只有句子和实体对但没有可用的关系标签，我们提供了基于远程监督的[关系标注工具](https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data)。

请您在使用前确认：

- 使用我们提供的三元组文件或确保您自定义的三元组文件质量较高
- 拥有足够的源数据
