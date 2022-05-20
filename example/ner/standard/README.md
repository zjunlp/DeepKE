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

## Use models quickly

We align the relationship type and the entity type of [DUIE](https://ai.baidu.com/broad/download?dataset=dureader) with the cnschema

On this basis, two models are trained based on 'Chinese Bert WwM' and 'Chinese Roberta WwM ext'. The super parameters used in the model are the given parameters.

You can download the [model](https://drive.google.com/drive/folders/1zA8Ichx9nzU3GD92ptdyR_nmARB_7ovg) directly for experiments.You just need to change the file name of the download folder to 'checkpoints' and then you can use it easily.

For example,input a sentence '《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽', DeepKE-chschema will output the result that the entity type of "星空黑夜传奇" is "网络小说" and "起点中文网" is "网站" with cnschema aligning.

## Model

BERT