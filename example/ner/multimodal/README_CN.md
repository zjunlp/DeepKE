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
cd DeepKE/example/ner/multimodal
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 模型采用的数据是Twitter15和Twitter17，文本数据采用conll格式，更多信息可参考[UMT](https://github.com/jefferyYu/UMT/)

- 存放数据： 可先下载数据 ```wget 120.27.214.45/Data/ner/multimodal/data.tar.gz```在此目录下

- Twitter15包含以下数据：

    - `twitter15_detect`：使用RCNN检测子图
    - `twitter2015_aux_images`：使用visual grouding检测的子图

    - `twitter2015_images`： 原始图片

    - `train.txt`: 训练文本数据

    - `...`

- 开始训练：模型加载和保存位置以及配置可以在conf的`.yaml`文件中修改
  
  - `python run.py` 

  - 训练好的模型默认保存在`checkpoint`中，可以通过修改`train.yaml`中的"save_path"更改保存路径

- 从上次训练的模型开始训练：设置`.yaml`中的save_path为上次保存模型的路径

- 每次训练的日志保存路径默认保存在当前目录，可以通过`.yaml`中的log_dir来配置

- 进行预测： 修改`predict.yaml`中的load_path来加载训练好的模型

- `python predict.py `

