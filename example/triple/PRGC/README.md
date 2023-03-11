# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/triple/PRGC/README_CN.md">简体中文</a> </b>
</p>

## Requirements

> python == 3.8

- torch == 1.5
- hydra-core == 1.0.6
- tensorboard == 2.4.1
- matplotlib == 3.4.1
- scikit-learn == 0.24.1
- transformers == 3.4.0
- jieba == 0.42.1
- wandb == 0.13.9
- deepke 

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/triple/PRGC
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    ```bash
    wget 120.27.214.45/Data/triple/PRGC/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - The dataset [CMeIE](https://tianchi.aliyun.com/dataset/95414)/ [NYT](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view)/ [NYT*](https://github.com/weizhepei/CasRel/tree/master/data/NYT)/ [WebNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)/ [WebNLG*](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG) is stored in `data`:
    - `rel2id.json`：Relation Labels / Answer words - ID

    - `test_triples.json`： Test set

    - `train_triples.json`: Training set

    - `val_triples.json`：Validation set
  
- Get pre-trained BERT model for PyTorch
  - Download [BERT-Base-Cased](https://huggingface.co/bert-base-cased)/ [BERT-Base-chinese](https://huggingface.co/bert-base-chinese) which contains pytroch_model.bin, vocab.txt and config.json. Put these under ./pretrain_models.
  - Rename config.json to bert_config.json
  - Replace '-' in folder name with '_' 

- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.

    ```bash
    python run.py
    ```

  - The trained model is stored in ./model directory by default.

  - Logs for training are stored in the ./logs directory by default.

- Prediction

  ```bash
  python predict.py
  ```


