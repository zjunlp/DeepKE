## 快速上手


<b> <a href="./README.md">English</a> | 简体中文 </b>


## 模型

<div align=center>
<img src="PURE.png" width="75%" height="75%" />
</div>

实体关系抽取模型**PURE** (NAACL'21) (详情请查阅论文[A Frustratingly Easy Approach for Entity and Relation Extraction](https://aclanthology.org/2021.naacl-main.5.pdf)).

### 环境依赖

> python == 3.8

- torch<1.13.0
- torchvision<0.14.0
- torchaudio<0.13.0
- tqdm==4.62.0
- allennlp==2.10.1
- transformers==4.20.0
- wandb==0.12.7
- hydra-core==1.3.1
- overrides
- requests

如果你想使用更高版本的transformers,如transformers 4.26.0，可以下载[allennlp](https://github.com/allenai/allennlp)源码放在example/triple/PURE目录下。
更改models目录下的entityModels.py和relationModels.py的import即可。下面是所需依赖，可自行复制至requirement.txt文件进行下载。

> python == 3.8
- torch<1.13.0
- torchvision<0.14.0
- tqdm==4.62.0
- transformers==4.26.0
- wandb==0.13.9
- hydra-core==1.3.1
- huggingface_hub==0.11.1
- overrides==7.3.1
- requests==2.28.2
- dill>=0.3.4
- base58>=2.1.1
- more_itertools>=8.12.0
- cached-path>=1.1.3
- protobuf==3.19.5
- spacy>=2.1.0
- fairscale==0.4.6
- jsonnet>=0.10.0 ; sys.platform != 'win32'
- nltk>=3.6.5
- numpy>=1.21.4
- tensorboardX>=1.2
- requests>=2.28
- tqdm>=4.62
- h5py>=3.6.0
- scikit-learn>=1.0.1
- scipy>=1.7.3
- pytest>=6.2.5
- sentencepiece>=0.1.96
- dataclasses;python_version<'3.7'
- filelock>=3.3
- lmdb>=1.2.1
- termcolor==1.1.0

### 克隆代码

```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/triple/PURE
```

### 使用pip安装

- 首先创建python虚拟环境，再进入虚拟环境

- 如果cuda版本是11.6, 请安装 torch==1.12.0+cu116: `pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116`.
- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据： 可先下载数据在`data`目录下

  ```bas
  cd data/
  wget 120.27.214.45/Data/triple/PURE/CMeIE.zip
  unzip CMeIE.zip
  ```

  支持一种类型文件格式：json格式，详细可参考文件夹。在 `data/CMeIE/json` 文件夹下存放训练数据：

  - `train.json`：存放训练数据集
  - `dev.json`：存放验证数据集
  - `test.json`：存放测试数据集

  在src/deepke/triple_extraction/PURE/models/const.py中存放关系种类

- 开始训练：```python run.py``` (训练所用到参数都在conf文件夹中，该任务支持多卡训练，可以修改`trian.yaml`中的参数以达到自己想要的效果。你还可以下载预训练实体模型或关系模型到`pretrain_models`文件夹中。

- 每次训练的日志和模型权重保存在`pretrain_models`文件夹内。

- 进行预测 ```python predict.py``` (预测结果将保存在conf/train.yaml设置的`relation_output_dir`参数中，默认为`pretrain_models/rel-cmeie-ctx0/`)

## 模型内容

PURE（基于论文 "[A Frustratingly Easy Approach for Entity and Relation Extraction](https://arxiv.org/pdf/2010.12812.pdf)")

## Train.yaml 参数

一些重要的参数如下所示，relation与entity模型参数类似:

- data_dir: 预处理数据集的路径

`实体模型参数:`

- entity_do_train: 是否进行训练
- entity_do_eval: 是否进行评估
- entity_eval_test: 是否在测试集上评估 (如果entity_do_eval为true且entity_eval_test为false，则在开发集上评估)
- entity_single_card: 是否使用单卡训练（默认为False）
- entity_learning_rate: BERT encoder的学习率
- entity_task_learning_rate: 任务特定参数的学习率.
- entity_train_batch_size: 训练的批量
- entity_num_epoch: 训练周期数
- entity_context_window: 实体模型的上下文窗口大小
- entity_model: 基本模型名称 (a huggingface model) (请保持relation_model和entity_model一致) 
- entity_output_dir: 实体模型的输出目录（包括日志、训练模型、实体预测结果等）
- entity_test_pred_filename: test set的预测文件名
- entity_dev_pred_filename: dev set的预测文件名

`关系模型参数:`

- no_cuda: 是否在可使用时不使用CUDA
- relation_train_file: 训练数据的路径
- relation_single_card: 是否使用单卡训练（默认为False）
- relation_do_lower_case: 如果使用的是uncased model，请设置此标志
- relation_train_batch_size: 训练的总批量
- relation_eval_batch_size: 评估的总批量大小
- relation_learning_rate: Adam的初始学习率
- relation_num_train_epochs: 要执行的训练总周期数
- relation_max_seq_length: 最大总输入序列长度。大于此长度的序列将被截断，小于此长度的将被填充。
- relation_prediction_file: 关系预测结果文件名
- relation_output_dir: 将在其中写入模型预测和检查点的输出目录

## 引用

```bibtex
@inproceedings{DBLP:conf/naacl/ZhongC21,
  author    = {Zexuan Zhong and
               Danqi Chen},
  editor    = {Kristina Toutanova and
               Anna Rumshisky and
               Luke Zettlemoyer and
               Dilek Hakkani{-}T{\"{u}}r and
               Iz Beltagy and
               Steven Bethard and
               Ryan Cotterell and
               Tanmoy Chakraborty and
               Yichao Zhou},
  title     = {A Frustratingly Easy Approach for Entity and Relation Extraction},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2021, Online, June 6-11, 2021},
  pages     = {50--61},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.naacl-main.5},
  doi       = {10.18653/v1/2021.naacl-main.5},
  timestamp = {Fri, 06 Aug 2021 00:41:31 +0200},
  biburl    = {https://dblp.org/rec/conf/naacl/ZhongC21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
