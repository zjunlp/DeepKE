# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/few-shot/README_CN.md">简体中文</a> </b>
</p>

## Requirements

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/few-shot
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    ```bash
    wget 120.27.214.45/Data/re/few-shot/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - The dataset [SEMEVAL](https://semeval2.fbk.eu/semeval2.php?location=tasks#T11) is stored in `data`:
    - `rel2id.json`：Relation Label - ID
    - `temp.txt`：Results of handled relation labels

    - `test.txt`： Test set

    - `train.txt`: Training set

    - `val.txt`：Validation set

- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.

  - Few-shot training on SEMEVAL

    ```bash
    python run.py
    ```

  - The trained model is stored in the current directory by default.

  - Start to train from last-trained model<br>

    modify `train_from_saved_model` in `.yaml` as the path of the last-trained model

  - Logs for training are stored in the current directory by default and the path can be configured by modifying `log_dir` in `.yaml`

- Prediction

  ```bash
  python predict.py
  ```

## Model

[KnowPrompt](https://arxiv.org/abs/2104.07650)
