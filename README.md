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
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/README_CN.md">简体中文</a> </b>
</p>

<h1 align="center">
    <p>A Deep Learning Based Knowledge Extraction Toolkit<br>for Knowledge Graph Construction</p>
</h1>


[DeepKE](https://arxiv.org/pdf/2201.03335.pdf) is a knowledge extraction toolkit for knowledge graph construction supporting **cnSchema**，**low-resource**, **document-level** and **multimodal** scenarios for *entity*, *relation* and *attribute* extraction. We provide [documents](https://zjunlp.github.io/DeepKE/), [online demo](http://deepke.zjukg.cn/), [paper](https://arxiv.org/pdf/2201.03335.pdf), [slides](https://drive.google.com/file/d/1IIeIZAbVduemqXc4zD40FUMoPHCJinLy/view?usp=sharing) and [poster](https://drive.google.com/file/d/1vd7xVHlWzoAxivN4T5qKrcqIGDcSM1_7/view?usp=sharing) for beginners.

- ❗Want to use Large Language Models with DeepKE? Try [DeepKE-LLM](https://github.com/zjunlp/DeepKE/tree/main/example/llm) with [KnowLM](https://github.com/zjunlp/KnowLM), have fun!
- ❗Want to train supervised models? Try [Quick Start](#quick-start), we provide the NER models (e.g, [LightNER(COLING'22)](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot), [W2NER(AAAI'22)](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/w2ner)), relation extraction models (e.g., [KnowPrompt(WWW'22)](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot)), relational triple extraction models (e.g., [ASP(EMNLP'22)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/ASP), [PRGC(ACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PRGC), [PURE(NAACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PURE)), and release off-the-shelf  models at [DeepKE-cnSchema](https://github.com/zjunlp/DeepKE/tree/main/example/triple/cnschema), have fun!

**If you encounter any issues during the installation of DeepKE and DeepKE-LLM, please check [Tips](https://github.com/zjunlp/DeepKE#tips) or promptly submit an [issue](https://github.com/zjunlp/DeepKE/issues), and we will assist you with resolving the problem!**


# Table of Contents

- [Table of Contents](#table-of-contents)
- [What's New](#whats-new)
- [Prediction Demo](#prediction-demo)
- [Model Framework](#model-framework)
- [Quick Start](#quick-start)
  - [DeepKE-LLM](#deepke-llm)
  - [DeepKE](#deepke)
      - [🔧Manual Environment Configuration](#manual-environment-configuration)
      - [🐳Building With Docker Images](#building-with-docker-images)
  - [Requirements](#requirements)
    - [DeepKE](#deepke-1)
  - [Introduction of Three Functions](#introduction-of-three-functions)
    - [1. Named Entity Recognition](#1-named-entity-recognition)
    - [2. Relation Extraction](#2-relation-extraction)
    - [3. Attribute Extraction](#3-attribute-extraction)
    - [4. Event Extraction](#4-event-extraction)
- [Tips](#tips)
- [To do](#to-do)
- [Reading Materials](#reading-materials)
- [Related Toolkit](#related-toolkit)
- [Citation](#citation)
- [Contributors (Determined by the roll of the dice)](#contributors-determined-by-the-roll-of-the-dice)
- [Other Knowledge Extraction Open-Source Projects](#other-knowledge-extraction-open-source-projects)

<br>

# What's New
* `Sep 2023` a bilingual Chinese English Information Extraction (IE) instruction dataset called  `InstructIE` was released for the Instruction based Knowledge Graph Construction Task (Instruction based KGC), as detailed in [here](./example/llm/README.md/#data).

* `June, 2023` We update [DeepKE-LLM](https://github.com/zjunlp/DeepKE/tree/main/example/llm) to support **knowledge extraction** with [KnowLM](https://github.com/zjunlp/KnowLM), [ChatGLM](https://github.com/THUDM/ChatGLM-6B), LLaMA-series, GPT-series etc.
* `Apr, 2023` We have added new models, including [CP-NER(IJCAI'23)](https://github.com/zjunlp/DeepKE/blob/main/example/ner/cross), [ASP(EMNLP'22)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/ASP), [PRGC(ACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PRGC), [PURE(NAACL'21)](https://github.com/zjunlp/DeepKE/tree/main/example/triple/PURE), provided [event extraction](https://github.com/zjunlp/DeepKE/tree/main/example/ee/standard) capabilities (Chinese and English), and offered compatibility with higher versions of Python packages (e.g., Transformers).

* `Feb, 2023` We have supported using [LLM](https://github.com/zjunlp/DeepKE/tree/main/example/llm) (GPT-3) with in-context learning (based on [EasyInstruct](https://github.com/zjunlp/EasyInstruct)) & data generation, added a NER model [W2NER(AAAI'22)](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/w2ner).

<details>
<summary><b>Previous News</b></summary>

* `Nov, 2022` Add data [annotation instructions](https://github.com/zjunlp/DeepKE/blob/main/README_TAG.md) for entity recognition and relation extraction, automatic labelling of weakly supervised data ([entity extraction](https://github.com/zjunlp/DeepKE/tree/main/example/ner/prepare-data) and [relation extraction](https://github.com/zjunlp/DeepKE/tree/main/example/re/prepare-data)), and optimize [multi-GPU training](https://github.com/zjunlp/DeepKE/tree/main/example/re/standard).
  
* `Sept, 2022` The paper [DeepKE: A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population](https://arxiv.org/abs/2201.03335) has been accepted by the EMNLP 2022 System Demonstration Track.

* `Aug, 2022` We have added [data augmentation](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot/DA) (Chinese, English) support for [low-resource relation extraction](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot).

* `June, 2022` We have added multimodal support for [entity](https://github.com/zjunlp/DeepKE/tree/main/example/ner/multimodal) and [relation extraction](https://github.com/zjunlp/DeepKE/tree/main/example/re/multimodal).

* `May, 2022` We have released [DeepKE-cnschema](https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA.md) with off-the-shelf knowledge extraction models.

* `Jan, 2022` We have released a paper [DeepKE: A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population](https://arxiv.org/abs/2201.03335)

* `Dec, 2021` We have added `dockerfile` to create the enviroment automatically. 

* `Nov, 2021` The demo of DeepKE, supporting real-time extration without deploying and training, has been released.
* The documentation of DeepKE, containing the details of DeepKE such as source codes and datasets, has been released.

* `Oct, 2021` `pip install deepke`
* The codes of deepke-v2.0 have been released.

* `Aug, 2019` The codes of deepke-v1.0 have been released.

* `Aug, 2018` The project DeepKE startup and codes of deepke-v0.1 have been released.
  

</details>

# Prediction Demo

There is a demonstration of prediction. The GIF file is created by [Terminalizer](https://github.com/faressoft/terminalizer). Get the [code](https://drive.google.com/file/d/1r4tWfAkpvynH3CBSgd-XG79rf-pB-KR3/view?usp=share_link).
<img src="pics/demo.gif" width="636" height="494" align=center>

<br>

# Model Framework

<h3 align="center">
    <img src="pics/architectures.png">
</h3>


- DeepKE contains a unified framework for **named entity recognition**, **relation extraction** and **attribute extraction**, the three  knowledge extraction functions.
- Each task can be implemented in different scenarios. For example, we can achieve relation extraction in **standard**, **low-resource (few-shot)**, **document-level** and **multimodal** settings.
- Each application scenario comprises of three components: **Data** including Tokenizer, Preprocessor and Loader, **Model** including Module, Encoder and Forwarder, **Core** including Training, Evaluation and Prediction. 

<br>

# Quick Start

## DeepKE-LLM

In the era of large models, DeepKE-LLM utilizes a completely new environment dependency.

```
conda create -n deepke-llm python=3.9
conda activate deepke-llm

cd example/llm
pip install -r requirements.txt
```

Please note that the `requirements.txt` file is located in the `example/llm` folder.

## DeepKE
- *DeepKE* supports `pip install deepke`. <br>Take the fully supervised relation extraction for example.
- *DeepKE* supports both **manual** and **docker image** environment configuration, you can choose the appropriate way to build.
#### 🔧Manual Environment Configuration

**Step1** Download the basic code

```bash
git clone --depth 1 https://github.com/zjunlp/DeepKE.git
```

**Step2** Create a virtual environment using `Anaconda` and enter it.<br>

```bash
conda create -n deepke python=3.8

conda activate deepke
```

1. Install *DeepKE* with source code

   ```bash
   pip install -r requirements.txt
   
   python setup.py install
   
   python setup.py develop
   ```

2. Install *DeepKE* with `pip`

   ```bash
   pip install deepke
   ```
   

**Step3** Enter the task directory

```bash
cd DeepKE/example/re/standard
```

**Step4** Download the dataset, or follow the [annotation instructions](https://github.com/zjunlp/DeepKE/blob/main/README_TAG.md) to obtain data

```bash
wget 120.27.214.45/Data/re/standard/data.tar.gz

tar -xzvf data.tar.gz
```

Many types of data formats are supported,and details are in each part. 

**Step5** Training (Parameters for training can be changed in the `conf` folder)

We support visual parameter tuning by using *[wandb](https://docs.wandb.ai/quickstart)*.

```bash
python run.py
```

**Step6** Prediction (Parameters for prediction can be changed in the `conf` folder)

Modify the path of the trained model in `predict.yaml`.The absolute path of the model needs to be used，such as `xxx/checkpoints/2019-12-03_ 17-35-30/cnn_ epoch21.pth`.

```bash
python predict.py
```

 - **❗NOTE: if you encounter any errors, please refer to the [Tips](#tips) or submit a GitHub issue.**



#### 🐳Building With Docker Images
**Step1** Install the Docker client

Install Docker and start the Docker service.

**Step2** Pull the docker image and run the container

```bash
docker pull zjunlp/deepke:latest
docker run -it zjunlp/deepke:latest /bin/bash
```

The remaining steps are the same as **Step 3 and onwards** in **Manual Environment Configuration**.

 - **❗NOTE: You can refer to the [Tips](#tips) to speed up installation**

## Requirements


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

## Introduction of Three Functions

### 1. Named Entity Recognition

- Named entity recognition seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, organizations, etc.

- The data is stored in `.txt` files. Some instances as following (Users can label data based on the tools [Doccano](https://github.com/doccano/doccano), [MarkTool](https://github.com/FXLP/MarkTool), or they can use the [Weak Supervision](https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data) with DeepKE to obtain data automatically):

  |                           Sentence                           |           Person           |    Location    |          Organization          |
  | :----------------------------------------------------------: | :------------------------: | :------------: | :----------------------------: |
  | 本报北京9月4日讯记者杨涌报道：部分省区人民日报宣传发行工作座谈会9月3日在4日在京举行。 |            杨涌            |      北京      |            人民日报            |
  | 《红楼梦》由王扶林导演，周汝昌、王蒙、周岭等多位专家参与制作。 | 王扶林，周汝昌，王蒙，周岭 |            |  |
  | 秦始皇兵马俑位于陕西省西安市,是世界八大奇迹之一。 |           秦始皇           | 陕西省，西安市 |                          |

- Read the detailed process in specific README
  - **[STANDARD (Fully Supervised)](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard)**
    
    ***We [support LLM](https://github.com/zjunlp/DeepKE/tree/main/example/llm) and provide the off-the-shelf model, [DeepKE-cnSchema-NER](https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA_CN.md), which will extract entities in cnSchema without training.***

    **Step1** Enter  `DeepKE/example/ner/standard`.  Download the dataset.

    ```bash
    wget 120.27.214.45/Data/ner/standard/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    **Step2** Training<br>

    The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.
  
    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```
  
  - **[FEW-SHOT](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot)**

    **Step1** Enter  `DeepKE/example/ner/few-shot`.  Download the dataset.

    ```bash
    wget 120.27.214.45/Data/ner/few_shot/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```
  
    **Step2** Training in the low-resouce setting <br>
  
    The directory where the model is loaded and saved and the configuration parameters can be cusomized in the `conf` folder.
  
    ```bash
    python run.py +train=few_shot
    ```
    
    Users can modify `load_path` in `conf/train/few_shot.yaml` to use existing loaded model.<br>
    
    **Step3** Add `- predict` to `conf/config.yaml`, modify `loda_path` as the model path and `write_path` as the path where the predicted results are saved in `conf/predict.yaml`, and then run `python predict.py`
    
    ```bash
    python predict.py
    ```

  - **[MULTIMODAL](https://github.com/zjunlp/DeepKE/tree/main/example/ner/multimodal)**

    **Step1** Enter  `DeepKE/example/ner/multimodal`.  Download the dataset.

    ```bash
    wget 120.27.214.45/Data/ner/multimodal/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    We use RCNN detected objects and visual grounding objects from original images as visual local information, where RCNN via [faster_rcnn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py) and visual grounding via [onestage_grounding](https://github.com/zyang-ur/onestage_grounding).

    **Step2** Training in the multimodal setting <br>

    - The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.
    - Start with the model trained last time: modify `load_path` in `conf/train.yaml`as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by `log_dir`.

    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```

### 2. Relation Extraction

- Relationship extraction is the task of extracting semantic relations between entities from a unstructured text.

- The data is stored in `.csv` files. Some instances as following (Users can label data based on the tools [Doccano](https://github.com/doccano/doccano), [MarkTool](https://github.com/FXLP/MarkTool), or they can use the [Weak Supervision](https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data) with DeepKE to obtain data automatically):

  |                        Sentence                        | Relation |    Head    | Head_offset |    Tail    | Tail_offset |
  | :----------------------------------------------------: | :------: | :--------: | :---------: | :--------: | :---------: |
  | 《岳父也是爹》是王军执导的电视剧，由马恩然、范明主演。 |   导演   | 岳父也是爹 |      1      |    王军    |      8      |
  |  《九玄珠》是在纵横中文网连载的一部小说，作者是龙马。  | 连载网站 |   九玄珠   |      1      | 纵横中文网 |      7      |
  |     提起杭州的美景，西湖总是第一个映入脑海的词语。     | 所在城市 |    西湖    |      8      |    杭州    |      2      |

- **!NOTE: If there are multiple entity types for one relation, entity types can be prefixed with the relation as inputs.**
- Read the detailed process in specific README

  - **[STANDARD (Fully Supervised)](https://github.com/zjunlp/DeepKE/tree/main/example/re/standard)** 

    ***We [support LLM](https://github.com/zjunlp/DeepKE/tree/main/example/llm) and provide the off-the-shelf model, [DeepKE-cnSchema-RE](https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA_CN.md), which will extract relations in cnSchema without training.***

    **Step1** Enter the `DeepKE/example/re/standard` folder.  Download the dataset.

    ```bash
    wget 120.27.214.45/Data/re/standard/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    **Step2** Training<br>

    The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.
  
    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```
  
  - **[FEW-SHOT](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot)**

    **Step1** Enter `DeepKE/example/re/few-shot`. Download the dataset.

    ```bash
    wget 120.27.214.45/Data/re/few_shot/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    **Step 2** Training<br>

    - The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.
    - Start with the model trained last time: modify `train_from_saved_model` in `conf/train.yaml`as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by `log_dir`. 
  
    ```bash
    python run.py
    ```
  
    **Step3** Prediction
  
    ```bash
    python predict.py
    ```
  
  - **[DOCUMENT](https://github.com/zjunlp/DeepKE/tree/main/example/re/document)**<br>
  
    **Step1** Enter `DeepKE/example/re/document`.  Download the dataset.
  
    ```bash
    wget 120.27.214.45/Data/re/document/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```
    
    **Step2** Training<br>
  
    - The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.
    - Start with the model trained last time: modify `train_from_saved_model` in `conf/train.yaml`as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by `log_dir`. 
    
    ```bash
    python run.py
    ```
    
    **Step3** Prediction
    
    ```bash
    python predict.py
    ```

  - **[MULTIMODAL](https://github.com/zjunlp/DeepKE/tree/main/example/re/multimodal)**

    **Step1** Enter  `DeepKE/example/re/multimodal`.  Download the dataset.

    ```bash
    wget 120.27.214.45/Data/re/multimodal/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    We use RCNN detected objects and visual grounding objects from original images as visual local information, where RCNN via [faster_rcnn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py) and visual grounding via [onestage_grounding](https://github.com/zyang-ur/onestage_grounding).

    **Step2** Training<br>

    - The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.
    - Start with the model trained last time: modify `load_path` in `conf/train.yaml`as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by `log_dir`.

    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```

### 3. Attribute Extraction

- Attribute extraction is to extract attributes for entities in a unstructed text.

- The data is stored in `.csv` files. Some instances as following:

  |                           Sentence                           |   Att    |   Ent    | Ent_offset |      Val      | Val_offset |
  | :----------------------------------------------------------: | :------: | :------: | :--------: | :-----------: | :--------: |
  |          张冬梅，女，汉族，1968年2月生，河南淇县人           |   民族   |  张冬梅  |     0      |     汉族      |     6      |
  |诸葛亮，字孔明，三国时期杰出的军事家、文学家、发明家。|   朝代   |   诸葛亮   |     0      |     三国时期      |     8     |
  |        2014年10月1日许鞍华执导的电影《黄金时代》上映         | 上映时间 | 黄金时代 |     19     | 2014年10月1日 |     0      |

- Read the detailed process in specific README
  - **[STANDARD (Fully Supervised)](https://github.com/zjunlp/DeepKE/tree/main/example/ae/standard)**

    **Step1** Enter the `DeepKE/example/ae/standard` folder. Download the dataset.

    ```bash
    wget 120.27.214.45/Data/ae/standard/data.tar.gz
    
    tar -xzvf data.tar.gz
    ```

    **Step2** Training<br>

    The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.
    
    ```bash
    python run.py
    ```
    
    **Step3** Prediction
    
    ```bash
    python predict.py
    ```

<br>

### 4. Event Extraction

* Event extraction is the task to extract event type, event trigger words, event arguments from a unstructed text.
* The data is stored in `.tsv` files, some instances are as follows:

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

* Read the detailed process in specific README

  * [STANDARD(Fully Supervised)](./example/ee/standard/README.md)

    **Step1** Enter the `DeepKE/example/ee/standard` folder. Download the dataset.

    ```bash
    wget 120.27.214.45/Data/ee/DuEE.zip
    unzip DuEE.zip
    ```

    **Step 2** Training

    The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.

    ```bash
    python run.py
    ```

    **Step 3** Prediction

    ```bash
    python predict.py
    ```

<br>

# Tips

1.```Using nearest mirror```, **[THU](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) in China, will speed up the installation of *Anaconda*; [aliyun](http://mirrors.aliyun.com/pypi/simple/) in China, will speed up `pip install XXX`**.

2.When encountering `ModuleNotFoundError: No module named 'past'`，run `pip install future` .

3.It's slow to install the pretrained language models online. Recommend download pretrained models before use and save them in the `pretrained` folder. Read `README.md` in every task directory to check the specific requirement for saving pretrained models.

4.The old version of *DeepKE* is in the [deepke-v1.0](https://github.com/zjunlp/DeepKE/tree/deepke-v1.0) branch. Users can change the branch to use the old version. The old version has been totally transfered to the standard relation extraction ([example/re/standard](https://github.com/zjunlp/DeepKE/blob/main/example/re/standard/README.md)).

5.If you want to modify the source code, it's recommended to install *DeepKE* with source codes. If not, the modification will not work. See [issue](https://github.com/zjunlp/DeepKE/issues/117)

6.More related low-resource knowledge extraction  works can be found in [Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective](https://arxiv.org/pdf/2202.08063.pdf).

7.Make sure the exact versions of requirements in `requirements.txt`.

# To do
In next version, we plan to release a stronger LLM for KE. 

Meanwhile, we will offer long-term maintenance to **fix bugs**, **solve issues** and meet **new requests**. So if you have any problems, please put issues to us.

# Reading Materials

Data-Efficient Knowledge Graph Construction, 高效知识图谱构建 ([Tutorial on CCKS 2022](http://sigkg.cn/ccks2022/?page_id=24)) \[[slides](https://drive.google.com/drive/folders/1xqeREw3dSiw-Y1rxLDx77r0hGUvHnuuE)\] 

Efficient and Robust Knowledge Graph Construction ([Tutorial on AACL-IJCNLP 2022](https://www.aacl2022.org/Program/tutorials)) \[[slides](https://github.com/NLP-Tutorials/AACL-IJCNLP2022-KGC-Tutorial)\] 

PromptKG Family: a Gallery of Prompt Learning & KG-related Research Works, Toolkits, and Paper-list [[Resources](https://github.com/zjunlp/PromptKG)\] 

Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective \[[Survey](https://arxiv.org/abs/2202.08063)\]\[[Paper-list](https://github.com/zjunlp/Low-resource-KEPapers)\]


# Related Toolkit

[Doccano](https://github.com/doccano/doccano)、[MarkTool](https://github.com/FXLP/MarkTool)、[LabelStudio](https://labelstud.io/ ): Data Annotation Toolkits

[LambdaKG](https://github.com/zjunlp/PromptKG/tree/main/lambdaKG): A library and benchmark for PLM-based KG embeddings

[EasyInstruct](https://github.com/zjunlp/EasyInstruct): An easy-to-use framework to instruct Large Language Models

**Reading Materials**:

Data-Efficient Knowledge Graph Construction, 高效知识图谱构建 ([Tutorial on CCKS 2022](http://sigkg.cn/ccks2022/?page_id=24)) \[[slides](https://drive.google.com/drive/folders/1xqeREw3dSiw-Y1rxLDx77r0hGUvHnuuE)\] 

Efficient and Robust Knowledge Graph Construction ([Tutorial on AACL-IJCNLP 2022](https://www.aacl2022.org/Program/tutorials)) \[[slides](https://github.com/NLP-Tutorials/AACL-IJCNLP2022-KGC-Tutorial)\] 

PromptKG Family: a Gallery of Prompt Learning & KG-related Research Works, Toolkits, and Paper-list [[Resources](https://github.com/zjunlp/PromptKG)\] 

Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective \[[Survey](https://arxiv.org/abs/2202.08063)\]\[[Paper-list](https://github.com/zjunlp/Low-resource-KEPapers)\]


**Related Toolkit**:

[Doccano](https://github.com/doccano/doccano)、[MarkTool](https://github.com/FXLP/MarkTool)、[LabelStudio](https://labelstud.io/ ): Data Annotation Toolkits

[LambdaKG](https://github.com/zjunlp/PromptKG/tree/main/lambdaKG): A library and benchmark for PLM-based KG embeddings

[EasyInstruct](https://github.com/zjunlp/EasyInstruct): An easy-to-use framework to instruct Large Language Models

# Citation

Please cite our paper if you use DeepKE in your work

```bibtex
@inproceedings{EMNLP2022_Demo_DeepKE,
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
  title     = {DeepKE: {A} Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population},
  booktitle = {{EMNLP} (Demos)},
  pages     = {98--108},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.emnlp-demos.10}
}
```
<br>

# Contributors (Determined by the roll of the dice)

Zhejiang University: [Ningyu Zhang](https://person.zju.edu.cn/en/ningyu), Liankuan Tao, Xin Xu, Honghao Gui, Xiaohan Wang, Zekun Xi, Xinrong Li, Haiyang Yu, Hongbin Ye, Shuofei Qiao, Peng Wang, Yuqi Zhu, Xin Xie, Xiang Chen, Zhoubo Li, Lei Li, Xiaozhuan Liang, Yunzhi Yao, Jing Chen, Yuqi Zhu, Shumin Deng, Wen Zhang, Guozhou Zheng, Huajun Chen

Community Contributors: [thredreams](https://github.com/thredreams), [eltociear](https://github.com/eltociear)

Alibaba Group: Feiyu Xiong, Qiang Chen

DAMO Academy: Zhenru Zhang, Chuanqi Tan, Fei Huang

Intern: Ziwen Xu, Rui Huang, Xiaolong Weng

# Other Knowledge Extraction Open-Source Projects

- [CogIE](https://github.com/jinzhuoran/CogIE)
- [OpenNRE](https://github.com/thunlp/OpenNRE)
- [OmniEvent](https://github.com/THU-KEG/OmniEvent)
- [OpenUE](https://github.com/zjunlp/OpenUE)
- [OpenIE](https://stanfordnlp.github.io/CoreNLP/openie.html)
- [RESIN](https://github.com/RESIN-KAIROS/RESIN-pipeline-public)
- [ZShot](https://github.com/IBM/zshot)
- [ZS4IE](https://github.com/BBN-E/ZS4IE)
- [OmniEvent](https://github.com/THU-KEG/OmniEvent)
