# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/document/README_CN.md">简体中文</a> </b>
</p>

## Requirements

> python == 3.8

- torch == 1.5.0
- transformers == 3.4.0
- opt-einsum == 3.3.0
- ujson
- deepke

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/document
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    ```bash
    wget 121.41.117.246:8080/Data/re/document/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - The dataset [DocRED](https://github.com/thunlp/DocRED/tree/master/) is stored in `data`:

    - `dev.json`：Validation set
    - `rel_info.json`：Relation set

    - `rel2id.json`：Relation labels - ID

    - `test.json`：Test set

    - `train_annotated.json`：Training set annotated manually

    - `train_distant.json`: Training set generated by distant supervision

- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.

  - Training on DocRED

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

  - After prediction, generated `result.json` is stored in the current directory

## Model

[DocuNet](https://arxiv.org/abs/2106.03618)