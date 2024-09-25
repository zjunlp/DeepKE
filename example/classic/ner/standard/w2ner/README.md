# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/w2ner/README_CN.md">简体中文</a> </b>
</p>

### Model

An entity recognition method for multiple scenarios based on **W2NER** (AAAI2022) (for details, please refer to the paper [Unified Named Entity Recognition as Word-Word Relation Classification](https://arxiv.org/pdf/2112.10070.pdf)).


<div align=center>
<img src="img/w2ner-model.png" width="100%" height="100%" />
</div>

Named entity recognition (NER) involves three main types, including planar, overlapping (aka nested), and discontinuous
NER, which are mostly studied separately. Recently, there has been increasing interest in unifying NER, `W2NER` uses
one model to simultaneously handle the above three tasks. Specifically, it models unified NER as a word-word relational
classification, a novel alternative. The architecture effectively models the neighbor relationship between entity 
words and Next-Neighboring-Word (NNW) and Tail-Head-Word-* (THW-*, where * stands for label) relations.

### Result
| Dataset   | MSRA  | People's Daily |
|-----------|-------|----------------|
| Precision | 96.33 | 96.76          |                                                                                            |
| Recall    | 95.49 | 96.11          |                                                                                                  |
| F1        | 95.91 | 96.43          |   

## Requirements

> python == 3.8 

- pytorch-transformers == 1.2.0
- torch == 1.5.0
- hydra-core == 1.0.6
- seqeval == 1.2.2
- tqdm == 4.60.0
- matplotlib == 3.4.1
- prettytable == 2.4.0
- numpy (1.21.4)
- pandas (1.3.4)
- deepke

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/standard/w2ner
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    ```bash
    wget 120.27.214.45/Data/ner/standard/data.tar.gz
    tar -xzvf data.tar.gz
    ```
  - By default, Chinese data sets are supported. If you want to use English data sets, you need to modify the lan
    in config.yaml before prediction, and install nltk, download nltk.download('punkt')
  - Three types of data formats are supported，including `json`,`docx` and `txt`. The dataset is stored in `data`：
    - `train.txt`: Training set
    - `valid.txt `: Validation set
    - `test.txt`: Test set

- Training

  - Parameters for training are in the `conf` folder and users can modify them before training.

  - Logs for training are in the `log` folder and the trained model is saved in the `checkpoints` folder.
  - W2NER hyperparameters are set in model.yaml, the parameters used in training are all in the conf folder, just modify it. Among them, `device` is the number of the specified GPU. If there is only a single card GPU, set to 0.
  ```bash
  python run.py
  ```

- Prediction
    
   Chinese datasets are supported by default. If English datasets are used, 'nltk' need to be installed and download the corresponding vocabulary by running 'nltk.download('punkt')'. **Meanwhile before prediction, 'lan' in *config.yaml* also need to be set *en*.**

  ```bash
  python predict.py
  ```
  
>`DeprecationWarning: In future, it will be an error for np.bool_ ...` encountered during prediction will not have any 
> effect on the results of your experiments. The next `Release` version of `DeepKE` will update the dependent packages 
> and methods, and this `Warning` will be resolved at that time.

## Prepare weak_supervised data

If you only have text data and corresponding dictionaries, but no canonical training data.

You can get weakly supervised formatted training data through automated labeling methods.

Please make sure that:

- Provide high-quality dictionaries
- Enough text data

<p align="left">
<a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data/README.md">prepare-data</a> </b>
</p>


## Cite

```bibtex
@inproceedings{DBLP:conf/aaai/Li00WZTJL22,
  author    = {Jingye Li and
               Hao Fei and
               Jiang Liu and
               Shengqiong Wu and
               Meishan Zhang and
               Chong Teng and
               Donghong Ji and
               Fei Li},
  title     = {Unified Named Entity Recognition as Word-Word Relation Classification},
  booktitle = {Thirty-Sixth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2022, Thirty-Fourth Conference on Innovative Applications of Artificial
               Intelligence, {IAAI} 2022, The Twelveth Symposium on Educational Advances
               in Artificial Intelligence, {EAAI} 2022 Virtual Event, February 22
               - March 1, 2022},
  pages     = {10965--10973},
  publisher = {{AAAI} Press},
  year      = {2022},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/21344},
  timestamp = {Tue, 12 Jul 2022 14:14:21 +0200},
  biburl    = {https://dblp.org/rec/conf/aaai/Li00WZTJL22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
