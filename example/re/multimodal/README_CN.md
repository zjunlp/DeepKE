## 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/multimodal/README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

### 克隆代码
```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/multimodal
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 模型采用的数据是[MRE](https://github.com/thecharm/Mega)，MRE数据集来自于[Multimodal Relation Extraction with Efficient Graph Alignment](https://dl.acm.org/doi/10.1145/3474085.3476968)

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/re/multimodal/data.tar.gz```在此目录下

- MRE包含以下数据：

    - `img_detect`：使用RCNN检测的实体
    - `img_vg`：使用visual grouding检测的实体

    - `img_org`： 原图像

    - `txt`: 文本你数据

    - `vg_data`：绑定原图和`img_vg`

    - `ours_rel2id.json` 关系集合

- 开始训练：模型加载和保存位置以及配置可以在conf的`.yaml`文件中修改
  
  - `python run.py` 

  - 训练好的模型默认保存在`checkpoint`中，可以通过修改`train.yaml`中的"save_path"更改保存路径

- 从上次训练的模型开始训练：设置`.yaml`中的save_path为上次保存模型的路径

- 每次训练的日志保存路径默认保存在当前目录，可以通过`.yaml`中的log_dir来配置

- 进行预测： 修改`predict.yaml`中的load_path来加载训练好的模型

- `python predict.py `


## 模型内容

IFAformer是一个基于Transformer的双流多模态模型
设计有隐式特征对齐，可用于多模态RE任务，它在视觉和文本中统一利用 Transformer 结构，而无需显式设计模态对齐结构
