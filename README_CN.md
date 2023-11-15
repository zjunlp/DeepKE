<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="pics/logo.png" width="400"/></a>
<p>
<p align="center">  
    <a href="http://deepke.zjukg.cn">
        <img alt="Documentation" src="https://img.shields.io/badge/demo-website-blue">
    </a>
    <a href="https://pypi.org/project/deepke/#files">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/deepke">
    </a>
    <a href="https://github.com/zjunlp/DeepKE/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/zjunlp/deepke">
    </a>
    <a href="http://zjunlp.github.io/DeepKE">
        <img alt="Documentation" src="https://img.shields.io/badge/doc-website-red">
    </a>
    <a href="https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>

<p align="center">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/README.md">English</a> | 简体中文 </b>
</p>


<h1 align="center">
    <p>基于深度学习的开源中文知识图谱抽取框架</p>
</h1>


[DeepKE](https://arxiv.org/pdf/2201.03335.pdf) 是一个开源的知识图谱抽取与构建工具，支持<b>cnSchema、低资源、长篇章、多模态</b>的知识抽取工具，可以基于<b>PyTorch</b>实现<b>命名实体识别</b>、<b>关系抽取</b>和<b>属性抽取</b>功能。同时为初学者提供了[文档](https://zjunlp.github.io/DeepKE/)，[在线演示](http://deepke.zjukg.cn/CN/index.html), [论文](https://arxiv.org/pdf/2201.03335.pdf), [演示文稿](https://github.com/zjunlp/DeepKE/blob/main/docs/slides/Slides-DeepKE-cn.pdf)和[海报](https://drive.google.com/file/d/1vd7xVHlWzoAxivN4T5qKrcqIGDcSM1_7/view)。

- ❗想用大模型做抽取吗？试试[DeepKE-LLM](https://github.com/zjunlp/DeepKE/tree/main/example/llm/README_CN.md)和[KnowLM](https://github.com/zjunlp/KnowLM)！
- ❗想自己全监督训抽取模型吗？试试[快速上手](#快速上手), 我们提供实体识别模型 (例如[LightNER(COLING'22)](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot/README_CN.md), [W2NER(AAAI'22)](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/w2ner/README_CN.md))、关系抽取模型(例如[KnowPrompt(WWW'22)](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot/README_CN.md))、实体关系联合抽取模型(例如[ASP(EMNLP'22)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/ASP/README_CN.md), [PRGC(ACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PRGC/README_CN.md), [PURE(NAACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PURE/README_CN.md)), 和基于cnSchema的开箱即用模型[DeepKE-cnSchema](https://github.com/zjunlp/DeepKE/tree/main/example/triple/cnschema/README_CN.md)！

**如果您在安装DeepKE和DeepKE-LLM中遇到任何问题（一般是包的版本兼容性问题）不用心急，您可以查阅[常见问题](https://github.com/zjunlp/DeepKE/blob/main/README_CN.md#%E5%A4%87%E6%B3%A8%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)或直接提[Issue](https://github.com/zjunlp/DeepKE/issues)，我们会尽全力帮助您解决问题**！

# 目录

- [目录](#目录)
- [新版特性](#新版特性)
- [预测演示](#预测演示)
- [模型架构](#模型架构)
- [快速上手](#快速上手)
  - [DeepKE-LLM](#deepke-llm)
  - [DeepKE](#deepke)
      - [🔧 手动环境部署](#-手动环境部署)
      - [🐳 基于容器部署](#-基于容器部署)
  - [环境依赖](#环境依赖)
    - [DeepKE](#deepke-1)
  - [具体功能介绍](#具体功能介绍)
    - [1. 命名实体识别NER](#1-命名实体识别ner)
    - [2. 关系抽取RE](#2-关系抽取re)
    - [3. 属性抽取AE](#3-属性抽取ae)
    - [4.事件抽取](#4事件抽取)
- [备注（常见问题）](#备注常见问题)
- [未来计划](#未来计划)
- [阅读资料](#阅读资料)
- [相关工具](#相关工具)
- [引用](#引用)
- [项目贡献人员 （排名不分先后）](#项目贡献人员-排名不分先后)
- [其它知识抽取开源工具](#其它知识抽取开源工具)

<br>

# 新版特性

* `2023年9月` 为基于指令的知识图谱构建任务(Instruction-based KGC)发布了一个中英双语信息抽取(IE)指令数据集 `InstructIE`, 具体参见[此处](./example/llm/README_CN.md/#数据)。

* `2023年6月` 为[DeepKE-LLM](https://github.com/zjunlp/DeepKE/tree/main/example/llm)新增多个大模型(如[ChatGLM](https://github.com/THUDM/ChatGLM-6B)、LLaMA系列、GPT系列、抽取大模型[智析](https://github.com/zjunlp/KnowLM))支持。
* `2023年4月` 新增实体关系抽取模型[CP-NER(IJCAI'23)](https://github.com/zjunlp/DeepKE/blob/main/example/ner/cross/README_CN.md), [ASP(EMNLP'22)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/ASP/README_CN.md), [PRGC(ACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PRGC/README_CN.md), [PURE(NAACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PURE/README_CN.md), 支持[事件抽取](https://github.com/zjunlp/DeepKE/blob/main/example/ee/standard/README_CN.md)(中文、英文), 提供对Python库高级版本的支持 (例如Transformers)。

* `2023年2月` 支持[大模型](https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md) (GPT-3)，包含In-context Learning (基于 [EasyInstruct](https://github.com/zjunlp/EasyInstruct))和数据生成，新增实体识别模型[W2NER(AAAI'22)](https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/README_CN.md)。

<details>
<summary><b>旧版新闻</b></summary>


- `2022年11月` 新增实体识别、关系抽取的[数据标注说明](https://github.com/zjunlp/DeepKE/blob/main/README_TAG_CN.md)和弱监督数据自动标注([实体识别](https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data/README_CN.md)、[关系抽取](https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data/README_CN.md))功能，优化[多GPU训练](https://github.com/zjunlp/DeepKE/blob/main/example/re/standard/README_CN.md)。

- `2022年9月` 论文 [DeepKE: A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population](https://arxiv.org/abs/2201.03335)被EMNLP2022 System Demonstration Track录用。

- `2022年8月` 新增针对[低资源关系抽取](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot)的[数据增强](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot/DA) (中文、英文)功能。


- `2022年6月` 新增支持多模态场景的[实体抽取](https://github.com/zjunlp/DeepKE/tree/main/example/ner/multimodal)、[关系抽取](https://github.com/zjunlp/DeepKE/tree/main/example/re/multimodal)功能。

- `2022年5月` 发布[DeepKE-cnschema](https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA_CN.md)特别版模型，支持基于cnSchema的开箱即用的中文实体识别和关系抽取。

- `2022年1月` 发布论文 [DeepKE: A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population](https://arxiv.org/abs/2201.03335)

- `2021年12月` 加入`dockerfile`以便自动创建环境

- `2021年11月` 发布DeepKE demo页面，支持实时抽取，无需部署和训练模型
- 发布DeepKE文档，包含DeepKE源码和数据集等详细信息

- `2021年10月` `pip install deepke`
- deepke-v2.0发布

- `2019年8月` `pip install deepke`
- deepke-v1.0发布

- `2018年8月` DeepKE项目启动，deepke-v0.1代码发布

</details>

# 预测演示
下面使用一个demo展示预测过程。该动图由[Terminalizer](https://github.com/faressoft/terminalizer)生成，生成[代码](https://drive.google.com/file/d/1r4tWfAkpvynH3CBSgd-XG79rf-pB-KR3/view?usp=share_link)可点击获取。
<img src="pics/demo.gif" width="636" height="494" align=center>

<br>

# 模型架构

Deepke的架构图如下所示

<h3 align="center">
    <img src="pics/architectures.png">
</h3>

- DeepKE为三个知识抽取功能（命名实体识别、关系抽取和属性抽取）设计了一个统一的框架
- 可以在不同场景下实现不同功能。比如，可以在标准全监督、低资源少样本、文档级和多模态设定下进行关系抽取
- 每一个应用场景由三个部分组成：Data部分包含Tokenizer、Preprocessor和Loader，Model部分包含Module、Encoder和Forwarder，Core部分包含Training、Evaluation和Prediction


<br>

# 快速上手

## DeepKE-LLM
大模型时代, DeepKE-LLM采用全新的环境依赖
```
conda create -n deepke-llm python=3.9
conda activate deepke-llm

cd example/llm
pip install -r requirements.txt
```
注意！！是example/llm文件夹下的 `requirements.txt`

## DeepKE
- DeepKE支持pip安装使用，下以常规关系抽取场景为例
- DeepKE支持手动环境部署与容器部署，您可任选一种方法进行安装
#### 🔧 手动环境部署
**Step 1**：下载代码 ```git clone --depth 1 https://github.com/zjunlp/DeepKE.git```（别忘记star和fork哈！！！）

**Step 2**：使用anaconda创建虚拟环境，进入虚拟环境（提供[Dockerfile](https://github.com/zjunlp/DeepKE/tree/main/docker)源码和[教程](https://github.com/zjunlp/DeepKE/issues/320)可自行创建镜像；可参考[备注（常见问题）](#备注常见问题)使用镜像加速）

```bash
conda create -n deepke python=3.8

conda activate deepke
```
1） 基于pip安装，直接使用

```bash
pip install deepke
```

2） 基于源码安装

```bash
pip install -r requirements.txt

python setup.py install

python setup.py develop
```

**Step 3** ：进入任务文件夹，以常规关系抽取为例

```
cd DeepKE/example/re/standard
```

**Step 4**：下载数据集，或根据[数据标注说明](https://github.com/zjunlp/DeepKE/blob/main/README_TAG_CN.md)标注数据
```
wget 120.27.214.45/Data/re/standard/data.tar.gz

tar -xzvf data.tar.gz
```

支持多种数据类型格式，具体请见各部分子README。

**Step 5** ：模型训练，训练用到的参数可在conf文件夹内修改

DeepKE使用[wandb](https://docs.wandb.ai/quickstart)支持可视化调参

```
python run.py
```

**Step 6** ：模型预测。预测用到的参数可在conf文件夹内修改

修改`conf/predict.yaml`中保存训练好的模型路径。需使用模型的绝对路径。如`xxx/checkpoints/2019-12-03_17-35-30/cnn_epoch21.pth`。
```
python predict.py
```
- **❗注意: 如果您在安装或使用过程中遇到任何问题，您可以查看[备注（常见问题）](#备注常见问题) 或提交 GitHub issue.**

#### 🐳 基于容器部署

**Step1** 下载Docker客户端

从官网下载Docker客户端并启动Docker服务

**Step2** 拉取镜像并运行容器

```bash
docker pull zjunlp/deepke:latest
docker run -it zjunlp/deepke:latest /bin/bash
```

剩余步骤同**手动环境部署**一节中的**Step 3**及后续步骤相同

 - **❗注意: 您可以参考 [Tips](#tips) 来加速您的部署**
<br>

## 环境依赖


### DeepKE

> python == 3.8

- torch>=1.5,<=1.11
- hydra-core==1.0.6
- tensorboard==2.4.1
- matplotlib==3.4.1
- transformers==4.26.0
- jieba==0.42.1
- scikit-learn==0.24.1
- seqeval==1.2.2
- opt-einsum==3.3.0
- wandb==0.12.7
- ujson==5.6.0
- huggingface_hub==0.11.0
- tensorboardX==2.5.1
- nltk==3.8
- protobuf==3.20.1
- numpy==1.21.0
- ipdb==0.13.11
- pytorch-crf==0.7.2
- tqdm==4.66.1
- openai==0.28.0
- Jinja2==3.1.2
- datasets==2.13.2
- pyhocon==0.3.60

<br>

## 具体功能介绍

### 1. 命名实体识别NER

- 命名实体识别是从非结构化的文本中识别出实体和其类型。数据为txt文件，样式范例为(用户可以基于工具[Doccano](https://github.com/doccano/doccano)、[MarkTool](https://github.com/FXLP/MarkTool)标注数据，也可以通过DeepKE自带的[弱监督功能](https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data/README_CN.md)自动得到数据)：

  |                           Sentence                           |           Person           |    Location    |          Organization          |
  | :----------------------------------------------------------: | :------------------------: | :------------: | :----------------------------: |
  | 本报北京9月4日讯记者杨涌报道：部分省区人民日报宣传发行工作座谈会9月3日在4日在京举行。 |            杨涌            |      北京      |            人民日报            |
  | 《红楼梦》由王扶林导演，周汝昌、王蒙、周岭等多位专家参与制作。 | 王扶林，周汝昌，王蒙，周岭 |            |  |
  | 秦始皇兵马俑位于陕西省西安市,是世界八大奇迹之一。 |           秦始皇           | 陕西省，西安市 |                          |

- 具体流程请进入详细的README中
  - **[常规全监督STANDARD](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard)**  
  
     ***我们还提供了[大模型支持](https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md)和开箱即用的[DeepKE-cnSchema特别版](https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA_CN.md)，无需训练即可抽取支持cnSchema的实体***
  
     **Step1**: 进入`DeepKE/example/ner/standard`，下载数据集
     
     ```bash
     wget 120.27.214.45/Data/ner/standard/data.tar.gz
     
     tar -xzvf data.tar.gz
     ```
     
     **Step2**: 模型训练<br>
     
     数据集和参数配置可以分别在`data`和`conf`文件夹中修改
     
     ```
     python run.py
     ```
     
     **Step3**: 模型预测
     ```
     python predict.py
     ```
     
  - **[少样本FEW-SHOT](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot)** 
  
    **Step1**: 进入`DeepKE/example/ner/few-shot`，下载数据集
    
    ```bash
    wget 120.27.214.45/Data/ner/few_shot/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```
    
    **Step2**：低资源场景下训练模型<br>
    
    模型加载和保存位置以及参数配置可以在`conf`文件夹中修改
    
     ```
     python run.py +train=few_shot
     ```
    
    若要加载模型，修改`few_shot.yaml`中的`load_path`；<br>
    
    **Step3**：在`config.yaml`中追加`- predict`，`predict.yaml`中修改`load_path`为模型路径以及`write_path`为预测结果的保存路径，完成修改后使用
    
    ```
    python predict.py
    ```

  - **[多模态](https://github.com/zjunlp/DeepKE/tree/main/example/ner/multimodal)**

    **Step1**: 进入 `DeepKE/example/ner/multimodal`， 下载数据集

    ```bash
    wget 120.27.214.45/Data/ner/multimodal/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    我们在原始图像上使用[faster_rcnn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)和[visual grounding工具](https://github.com/zyang-ur/onestage_grounding)分别抽取RCNN objects和visual grounding objects来作为局部视觉信息

    **Step2** 多模态场景下训练模型 <br>

    - 数据集和参数配置可以分别进入`data`和`conf`文件夹中修改
    - 如需从上次训练的模型开始训练：设置`conf/train.yaml`中的`load_path`为上次保存模型的路径，每次训练的日志默认保存在根目录，可用`log_dir`来配置

    ```bash
    python run.py
    ```

    **Step3** 模型预测

    ```bash
    python predict.py
    ```

### 2. 关系抽取RE

- 关系抽取是从非结构化的文本中抽取出实体之间的关系，以下为几个样式范例，数据为csv文件(用户可以基于工具[Doccano](https://github.com/doccano/doccano)、[MarkTool](https://github.com/FXLP/MarkTool)标注数据，也可以通过DeepKE自带的[弱监督功能](https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data/README_CN.md)自动得到数据)：

  |                        Sentence                        | Relation |    Head    | Head_offset |    Tail    | Tail_offset |
  | :----------------------------------------------------: | :------: | :--------: | :---------: | :--------: | :---------: |
  | 《岳父也是爹》是王军执导的电视剧，由马恩然、范明主演。 |   导演   | 岳父也是爹 |      1      |    王军    |      8      |
  |  《九玄珠》是在纵横中文网连载的一部小说，作者是龙马。  | 连载网站 |   九玄珠   |      1      | 纵横中文网 |      7      |
  |     提起杭州的美景，西湖总是第一个映入脑海的词语。     | 所在城市 |    西湖    |      8      |    杭州    |      2      |
  
- **❗NOTE: 如果您使用的同一个关系存在多种实体类型，可以采取对实体类型加关系前缀的方式构造输入。**

- 具体流程请进入详细的README中，RE包括了以下三个子功能
  - **[常规全监督STANDARD](https://github.com/zjunlp/DeepKE/tree/main/example/re/standard)**  

     ***我们还提供了[大模型支持](https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md)和开箱即用的[DeepKE-cnSchema特别版](https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA_CN.md)，无需训练即可抽取支持cnSchema的关系***

    **Step1**：进入`DeepKE/example/re/standard`，下载数据集
  
    ```bash
    wget 120.27.214.45/Data/re/standard/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```
  
    **Step2**：模型训练<br>

    数据集和参数配置可以分别进入`data`和`conf`文件夹中修改
  
    ```
    python run.py
    ```
  
    **Step3**：模型预测
  
    ```
    python predict.py
    ```
  
  - **[少样本FEW-SHOT](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot)**
  
    **Step1**：进入`DeepKE/example/re/few-shot`，下载数据集

    ```bash
    wget 120.27.214.45/Data/re/few_shot/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```
  
    **Step2**：模型训练<br>
  
    - 数据集和参数配置可以分别进入`data`和`conf`文件夹中修改
  
    - 如需从上次训练的模型开始训练：设置`conf/train.yaml`中的`train_from_saved_model`为上次保存模型的路径，每次训练的日志默认保存在根目录，可用`log_dir`来配置
  
    ```
    python run.py
    ```
  
    **Step3**：模型预测
  
    ```
    python predict.py
    ```
  
  - **[文档级DOCUMENT](https://github.com/zjunlp/DeepKE/tree/main/example/re/document)** <br>
    
    **Step1**：进入`DeepKE/example/re/document`，下载数据集
    
    ```bash
    wget 120.27.214.45/Data/re/document/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```
    
    **Step2**：模型训练<br>
    
    - 数据集和参数配置可以分别进入`data`和`conf`文件夹中修改
    - 如需从上次训练的模型开始训练：设置`conf/train.yaml`中的`train_from_saved_model`为上次保存模型的路径，每次训练的日志默认保存在根目录，可用`log_dir`来配置；
    
    ```
    python run.py
    ```
    **Step3**：模型预测
    
    ```
    python predict.py
    ```

  - **[多模态](https://github.com/zjunlp/DeepKE/tree/main/example/re/multimodal)**

    **Step1**: 进入 `DeepKE/example/re/multimodal`， 下载数据集

    ```bash
    wget 120.27.214.45/Data/re/multimodal/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    我们在原始图像上使用[faster_rcnn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)和[visual grounding工具](https://github.com/zyang-ur/onestage_grounding)分别抽取RCNN objects和visual grounding objects来作为局部视觉信息

    **Step2** 模型训练 <br>

    - 数据集和参数配置可以分别进入`data`和`conf`文件夹中修改
    - 如需从上次训练的模型开始训练：设置`conf/train.yaml`中的`load_path`为上次保存模型的路径，每次训练的日志默认保存在根目录，可用`log_dir`来配置

    ```bash
    python run.py
    ```

    **Step3** 模型预测

    ```bash
    python predict.py
    ```

### 3. 属性抽取AE

- 数据为csv文件，样式范例为：

  |                           Sentence                           |   Att    |   Ent    | Ent_offset |      Val      | Val_offset |
  | :----------------------------------------------------------: | :------: | :------: | :--------: | :-----------: | :--------: |
  |          张冬梅，女，汉族，1968年2月生，河南淇县人           |   民族   |  张冬梅  |     0      |     汉族      |     6      |
  | 诸葛亮，字孔明，三国时期杰出的军事家、文学家、发明家。 |   朝代   |   诸葛亮   |     0      |     三国时期      |     8     |
  |        2014年10月1日许鞍华执导的电影《黄金时代》上映         | 上映时间 | 黄金时代 |     19     | 2014年10月1日 |     0      |

- 具体流程请进入详细的README中
  - **[常规全监督STANDARD](https://github.com/zjunlp/DeepKE/tree/main/example/ae/standard)**  
    
    **Step1**：进入`DeepKE/example/ae/standard`，下载数据集
    
    ```bash
    wget 120.27.214.45/Data/ae/standard/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```
    
    **Step2**：模型训练<br>

    数据集和参数配置可以分别进入`data`和`conf`文件夹中修改
    
    ```
    python run.py
    ```
    
    **Step3**：模型预测
    
    ```
    python predict.py
    ```

<br>

### 4.事件抽取

* 事件抽取是指从一段无结构化的文本中抽取出某个事件的事件类型、事件触发词、论元角色以及论元。

* 数据为`.tsv`文件，样例为：

  <table h style="text-align:center">
      <tr>
          <th colspan="2"> Sentence </th>
          <th> Event type </th>
          <th> Trigger </th>
          <th> Role </th>
          <th> Argument </th>
      </tr>
      <tr> 
          <td rowspan="3" colspan="2"> 据《欧洲时报》报道，当地时间27日，法国巴黎卢浮宫博物馆员工因不满工作条件恶化而罢工，导致该博物馆也因此闭门谢客一天。 </td>
        	<td rowspan="3"> 组织行为-罢工 </td>
      		<td rowspan="3"> 罢工 </td>
      		<td> 罢工人员 </td>
      		<td> 法国巴黎卢浮宫博物馆员工 </td>
      </tr>
      <tr> 
          <td> 时间 </td>
          <td> 当地时间27日 </td>
      </tr>
      <tr> 
          <td> 所属组织 </td>
          <td> 法国巴黎卢浮宫博物馆 </td>
      </tr>
      <tr> 
          <td rowspan="3" colspan="2"> 中国外运2019年上半年归母净利润增长17%：收购了少数股东股权 </td>
        	<td rowspan="3"> 财经/交易-出售/收购 </td>
      		<td rowspan="3"> 收购 </td>
      		<td> 出售方 </td>
      		<td> 少数股东 </td>
      </tr>
      <tr> 
          <td> 收购方 </td>
          <td> 中国外运 </td>
      </tr>
      <tr> 
          <td> 交易物 </td>
          <td> 股权 </td>
      </tr>
      <tr> 
          <td rowspan="3" colspan="2"> 美国亚特兰大航展13日发生一起表演机坠机事故，飞行员弹射出舱并安全着陆，事故没有造成人员伤亡。 </td>
        	<td rowspan="3"> 灾害/意外-坠机 </td>
      		<td rowspan="3"> 坠机 </td>
      		<td> 时间 </td>
      		<td> 13日 </td>
      </tr>
      <tr> 
          <td> 地点 </td>
          <td> 美国亚特兰 </td>
    	</tr>
  </table>

- 具体流程请进入详细的README中

  - **[常规全监督STANDARD](./example/ee/standard/README_CN.md)**  

    **Step1**：进入`DeepKE/example/ee/standard`，下载数据集

    ```bash
    wget 120.27.214.45/Data/ee/DuEE.zip
    unzip DuEE.zip
    ```

    **Step2**：模型训练<br>

    数据集和参数配置可以分别进入`data`和`conf`文件夹中修改

    ```
    python run.py
    ```

    **Step3**：模型预测

    ```
    python predict.py
    ```

<br>

# 备注（常见问题）

1.使用 Anaconda 时，```建议添加国内镜像```，下载速度更快。如[镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

2.使用 pip 时，```建议使用国内镜像```，下载速度更快，如阿里云镜像。

3.安装后提示 `ModuleNotFoundError: No module named 'past'`，输入命令 `pip install future` 即可解决。

4.使用语言预训练模型时，在线安装下载模型比较慢，更建议提前下载好，存放到 pretrained 文件夹内。具体存放文件要求见文件夹内的 `README.md`。

5.DeepKE老版本位于[deepke-v1.0](https://github.com/zjunlp/DeepKE/tree/deepke-v1.0)分支，用户可切换分支使用老版本，老版本的能力已全部迁移到标准设定关系抽取([example/re/standard](https://github.com/zjunlp/DeepKE/blob/main/example/re/standard/README.md))中。

6.如果您需要在源码的基础上进行修改，建议使用`python setup.py install`方式安装*DeepKE*，如未使用该方式安装，源码修改部分不会生效，见[问题](https://github.com/zjunlp/DeepKE/issues/117)。

7.更多的低资源抽取工作可查阅论文 [Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective](https://arxiv.org/pdf/2202.08063.pdf)。

8.确保使用requirements.txt中对应的各依赖包的版本。

<br>

# 未来计划

- 在DeepKE的下一个版本中发布优化后的中英双语抽取大模型
- 我们提供长期技术维护和答疑解惑。如有疑问，请提交issues


# 阅读资料

Data-Efficient Knowledge Graph Construction, 高效知识图谱构建 ([Tutorial on CCKS 2022](http://sigkg.cn/ccks2022/?page_id=24)) \[[slides](https://pan.baidu.com/s/1yMskUVU188-4dcf96lVrWg?pwd=gy8y)\] 

Efficient and Robust Knowledge Graph Construction ([Tutorial on AACL-IJCNLP 2022](https://www.aacl2022.org/Program/tutorials)) \[[slides](https://github.com/NLP-Tutorials/AACL-IJCNLP2022-KGC-Tutorial)\] 

PromptKG Family: a Gallery of Prompt Learning & KG-related Research Works, Toolkits, and Paper-list [[Resources](https://github.com/zjunlp/PromptKG)\] 

Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective \[[Survey](https://arxiv.org/abs/2202.08063)\]\[[Paper-list](https://github.com/zjunlp/Low-resource-KEPapers)\]

基于大模型提示学习的推理工作综述 \[[论文](https://arxiv.org/abs/2212.09597)\]\[[列表](https://github.com/zjunlp/Prompt4ReasoningPapers)\]\[[ppt](https://github.com/zjunlp/Prompt4ReasoningPapers/blob/main/tutorial.pdf)\]

# 相关工具

[Doccano](https://github.com/doccano/doccano)、[MarkTool](https://github.com/FXLP/MarkTool)、[LabelStudio](https://labelstud.io/ )：实体识别关系抽取数据标注工具

[LambdaKG](https://github.com/zjunlp/PromptKG/tree/main/lambdaKG): 基于预训练语言模型的知识图谱表示与应用工具

[EasyInstruct](https://github.com/zjunlp/EasyInstruct): 一个基于指令使用大模型的工具


# 引用

如果使用DeepKE，请按以下格式引用

```bibtex
@inproceedings{DBLP:conf/emnlp/ZhangXTYYQXCLL22,
  author    = {Ningyu Zhang and
               Xin Xu and
               Liankuan Tao and
               Haiyang Yu and
               Hongbin Ye and
               Shuofei Qiao and
               Xin Xie and
               Xiang Chen and
               Zhoubo Li and
               Lei Li},
  editor    = {Wanxiang Che and
               Ekaterina Shutova},
  title     = {DeepKE: {A} Deep Learning Based Knowledge Extraction Toolkit for Knowledge
               Base Population},
  booktitle = {Proceedings of the The 2022 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2022 - System Demonstrations, Abu Dhabi,
               UAE, December 7-11, 2022},
  pages     = {98--108},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.emnlp-demos.10},
  timestamp = {Thu, 23 Mar 2023 16:56:00 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/ZhangXTYYQXCLL22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<br>

# 项目贡献人员 （排名不分先后）

浙江大学：[张宁豫](https://person.zju.edu.cn/ningyu)、陶联宽、徐欣、桂鸿浩、王潇寒、习泽坤、李欣荣、余海阳、叶宏彬、乔硕斐、王鹏、朱雨琦、谢辛、陈想、黎洲波、李磊、梁孝转、姚云志、陈静、朱雨琦、邓淑敏、张文、郑国轴、陈华钧

开源社区贡献者: [thredreams](https://github.com/thredreams), [eltociear](https://github.com/eltociear)

阿里巴巴：熊飞宇、陈强

阿里巴巴达摩院：张珍茹、谭传奇、黄非

实习生：徐子文、黄睿、翁晓龙

# 其它知识抽取开源工具

- [CogIE](https://github.com/jinzhuoran/CogIE)
- [OpenNRE](https://github.com/thunlp/OpenNRE)
- [OmniEvent](https://github.com/THU-KEG/OmniEvent)
- [OpenUE](https://github.com/zjunlp/OpenUE)
- [OpenIE](https://stanfordnlp.github.io/CoreNLP/openie.html)
- [RESIN](https://github.com/RESIN-KAIROS/RESIN-pipeline-public)
- [ZShot](https://github.com/IBM/zshot)
- [OmniEvent](https://github.com/THU-KEG/OmniEvent)
