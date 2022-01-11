# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/README_CN.md">简体中文</a> </b>
</p>

## Requirements

> python == 3.8 

- pytorch-transformers == 1.2.0
- torch == 1.5.0
- hydra-core == 1.0.6
- seqeval == 1.2.2
- tqdm == 4.60.0
- matplotlib == 3.4.1
- deepke

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/standard
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

  - The dataset is stored in `data`：
    - `train.txt`: Training set
    - `valid.txt `: Validation set
    - `test.txt`: Test set

- Training

  - Parameters for training are in the `conf` folder and users can modify them before training.

  - Logs for training are in the `log` folder and the trained model is saved in the `checkpoints` folder.

  ```bash
  python run.py
  ```

- Prediction

  ```bash
  python predict.py
  ```

## Model

BERT