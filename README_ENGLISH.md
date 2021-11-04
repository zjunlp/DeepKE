<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="pics/logo.png" width="400"/></a>
<p>
<p align="center">  
    <a href="https://deepke.openkg.cn">
        <img alt="Documentation" src="https://img.shields.io/badge/DeepKE-website-green">
    </a>
    <a href="https://pypi.org/project/deepke/#files">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/deepke">
    </a>
    <a href="https://github.com/zjunlp/DeepKE/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/zjunlp/deepke">
    </a>
</p>
<p align="center">
    <b><a href="https://github.com/zjunlp/DeepKE/blob/main/README.md">简体中文</a> | English</b>
</p>

<br>

<h2 align="center">
    <p>A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population</p>
</h2>

DeepKE is a knowledge extraction toolkit supporting **low-resource** and **document-level** scenarios. It provides three functions based **PyTorch**, including **Named Entity Recognition**, **Relation Extraciton** and **Attribute Extraction**.

<br>

## Online Demo

[demo](https://deepke.openkg.cn)

### Prediction

There is a demonstration of prediction.<br>
<img src="pics/demo.gif" width="636" height="494" align=center>

<br>

## Model Framework

<h3 align="center">
    <img src="pics/architectures.png">
</h3>
<p align="center">
    Figure 1: The framework of DeepKE
</p>

- DeepKE contains three modules for **named entity recognition**, **relation extraction** and **attribute extraction**, the three tasks respectively.
- Each module has its own submodules. For example, there are **standard**, **document-level** and **few-shot** submodules in the attribute extraction modular.
- Each submodule compose of three parts: a **collection of tools**, which can function as tokenizer, dataloader, preprocessor and the like, a **encoder** and a part for **training and prediction**

<br>

## Quickstart

*DeepKE* is supported `pip install deepke`. Take the fully supervised attribute extraction for example.

**Step1** Download basic codes `git clone https://github.com/zjunlp/DeepKE.git ` (Please star✨ and fork :memo:)

**Step2** Create a virtual environment using`Anaconda` and enter it

```bash
conda create -n deepke python=3.8

conda activate deepke
```

1. Install *DeepKE* with `pip`

   ```bash
   pip install deepke
   ```

2. Install *DeepKE* with source codes

   ```bash
   python setup.py install
   
   python setup.py develop
   ```

**Step3** Enter the task directory

```bash
cd DeepKE/example/re/standard
```

**Step4** Training (Parameters for training can be changed in the `conf` folder)

```bash
python run.py
```

**Step5** Prediction (Parameters for prediction can be changed in the `conf` folder)

```bash
python predict.py
```



### Requirements

> python == 3.8

- torch == 1.5
- hydra-core == 1.0.6
- tensorboard == 2.4.1
- matplotlib == 3.4.1
- transformers == 3.4.0
- jieba == 0.42.1
- scikit-learn == 0.24.1
- pytorch-transformers == 1.2.0
- seqeval == 1.2.2
- tqdm == 4.60.0
- opt-einsum==3.3.0
- ujson

### Introduction of Three Functions

#### 1. Named Entity Recognition

- Named entity recognition seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, organizations, etc.

- The data is stored in `.txt` files. Some instances as following:

  |                           Sentence                           |           Person           |    Location    |          Organization          |
  | :----------------------------------------------------------: | :------------------------: | :------------: | :----------------------------: |
  | 本报北京9月4日讯记者杨涌报道：部分省区人民日报宣传发行工作座谈会9月3日在4日在京举行。 |            杨涌            |      北京      |            人民日报            |
  | 《红楼梦》是中央电视台和中国电视剧制作中心根据中国古典文学名著《红楼梦》摄制于1987年的一部古装连续剧，由王扶林导演，周汝昌、王蒙、周岭等多位红学家参与制作。 | 王扶林，周汝昌，王蒙，周岭 |      中国      | 中央电视台，中国电视剧制作中心 |
  | 秦始皇兵马俑位于陕西省西安市，1961年被国务院公布为第一批全国重点文物保护单位，是世界八大奇迹之一。 |           秦始皇           | 陕西省，西安市 |             国务院             |

- Read the detailed process in specific README
  - **[STANDARD (Fully Supervised)](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/ner/standard)**

    **Step1** Enter  `DeepKE/example/ner/standard`. The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.<br>

    **Step2** Training

    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```

  - **[FEW-SHOT](https://github.com/zjunlp/DeepKE/tree/test_new_deepke/example/ner/few-shot)**

    **Step1** Enter  `DeepKE/example/ner/few-shot`. The directory where the model is loaded and saved and the configuration parameters can be cusomized in the `conf` folder.<br>

    **Step2** Training with default `CoNLL-2003` dataset.

    ```bash
    python run.py +train=few_shot
    ```

    Users can modify `load_path` in `conf/train/few_shot.yaml` with the use of existing loaded model.<br>

    **Step3** Add `- predict` to `conf/config.yaml`, modify `loda_path` as the model path and `write_path` as the path where the predicted results are saved in `conf/predict.yaml`, and then run `python predict.py`

    ```bash
    python predict.py
    ```

#### 2. Relation Extraction

- Relationship extraction is the task of extracting semantic relations between entities from a unstructured text.

- The data is stored in `.csv` files. Some instances as following:

  |                        Sentence                        | Relation |    Head    | Head_offset |    Tail    | Tail_offset |
  | :----------------------------------------------------: | :------: | :--------: | :---------: | :--------: | :---------: |
  | 《岳父也是爹》是王军执导的电视剧，由马恩然、范明主演。 |   导演   | 岳父也是爹 |      1      |    王军    |      8      |
  |  《九玄珠》是在纵横中文网连载的一部小说，作者是龙马。  | 连载网站 |   九玄珠   |      1      | 纵横中文网 |      7      |
  |     提起杭州的美景，西湖总是第一个映入脑海的词语。     | 所在城市 |    西湖    |      8      |    杭州    |      2      |

- Read the detailed process in specific README

  - **[STANDARD (Fully Supervised)](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/re/standard)** 

    **Step1** Enter the `DeepKE/example/re/standard` folder. The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.<br>

    **Step2** Training

    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```

  - **[FEW-SHOT](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/re/few-shot)**

    **Step1** Enter `DeepKE/example/re/few-shot`. The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.<br>

    **Step 2** Training. Start with the model trained last time: modify `train_from_saved_model` in `conf/train.yaml`as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by `log_dir`. <br>

    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```

  - **[DOCUMENT](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/re/document)**<br>

    Download the model `train_distant.json` from [*Google Drive*](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw) to `data/`.

    **Step1** Enter `DeepKE/example/re/document`. The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.<br>

    **Step2** Training. Start with the model trained last time: modify `train_from_saved_model` in `conf/train.yaml`as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by `log_dir`. 

    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```

#### 3. Attribute Extraction

- Attribute extraction is to extract attributes for entities in a unstructed text.

- The data is stored in `.csv` files. Some instances as following:

  |                           Sentence                           |   Att    |   Ent    | Ent_offset |      Val      | Val_offset |
  | :----------------------------------------------------------: | :------: | :------: | :--------: | :-----------: | :--------: |
  |          张冬梅，女，汉族，1968年2月生，河南淇县人           |   民族   |  张冬梅  |     0      |     汉族      |     6      |
  | 杨缨，字绵公，号钓溪，松溪县人，祖籍将乐，是北宋理学家杨时的七世孙 |   朝代   |   杨缨   |     0      |     北宋      |     22     |
  |        2014年10月1日许鞍华执导的电影《黄金时代》上映         | 上映时间 | 黄金时代 |     19     | 2014年10月1日 |     0      |

- Read the detailed process in specific README
  - **[STANDARD (Fully Supervised)](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/ae/standard)**

    **Step1** Enter the `DeepKE/example/ae/standard` folder. The dataset and parameters can be customized in the `data` folder and `conf` folder respectively.<br>

    **Step2** Training

    ```bash
    python run.py
    ```

    **Step3** Prediction

    ```bash
    python predict.py
    ```



## Notebook Tutorial

This toolkit provides many `Jupyter Notebook` and `Google Colab` tutorials. Users can study *DeepKE* with them.

- Standard Setting<br>

  [NER Notebook](https://github.com/zjunlp/DeepKE/blob/main/tutorial-notebooks/ner/standard/tutorial.ipynb)

  [NER Colab](https://colab.research.google.com/drive/1KpJFAT1nZfGDfnuNMZn02_okIU08j46d?usp=sharing)

  [RE Notebook](https://github.com/zjunlp/DeepKE/blob/main/tutorial-notebooks/re/standard/tutorial.ipynb)

  [RE Colab](https://colab.research.google.com/drive/1o6rKIxBqrGZNnA2IMXqiSsY2GWANAZLl?usp=sharing)

  [AE Notebook](https://github.com/zjunlp/DeepKE/blob/main/tutorial-notebooks/ae/standard/tutorial.ipynb)

  [AE Colab](https://colab.research.google.com/drive/1pgPouEtHMR7L9Z-QfG1sPYkJfrtRt8ML)

- Low-resource<br>

  [NER Notebook](https://github.com/zjunlp/DeepKE/blob/main/tutorial-notebooks/ner/few-shot/tutorial.ipynb)

  [NER Colab](https://colab.research.google.com/drive/1Xz0sNpYQNbkjhebCG5djrwM8Mj2Crj7F?usp=sharing)

  [RE Notebook](https://github.com/zjunlp/DeepKE/blob/main/tutorial-notebooks/re/few-shot/tutorial.ipynb)

  [RE Colab]()

- Document-level<br>

  [RE Notebook](https://github.com/zjunlp/DeepKE/blob/main/tutorial-notebooks/re/document/tutorial.ipynb)

  [RE Colab]()

<br>

## Tips

1. Using nearest mirror, like [THU](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) in China, will speed up the installation of *Anaconda*.
2. Using nearest mirror, like [aliyun](http://mirrors.aliyun.com/pypi/simple/) in China, will speed up `pip install XXX`.
3. When encountering `ModuleNotFoundError: No module named 'past'`，run `pip install future` .
4. It's slow to install the pretrained language models online. Recommend download pretrained models before use and save them in the `pretrained` folder. Read `README.md` in every task directory to check the specific requirement for saving pretrained models.

<br>

## Developers

Zhejiang University: Ningyu Zhang, Liankuan Tao, Haiyang Yu, Xiang Chen, Xin Xu, Xi Tian, Lei Li, Zhoubo Li, Shumin Deng, Yunzhi Yao, Hongbin Ye, Xin Xie, Guozhou Zheng, Huajun Chen

Alibaba DAMO: Chuanqi Tan, Fei Huang
