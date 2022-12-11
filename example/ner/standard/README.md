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

  - Three types of data formats are supported，including `json`,`docx` and `txt`. The dataset is stored in `data`：
    - `train.txt`: Training set
    - `valid.txt `: Validation set
    - `test.txt`: Test set

- Training

  - Parameters for training are in the `conf` folder and users can modify them before training.This task supports multi card training. Modify trian.yaml's parameter 'use_multi_gpu' to true.'os.environ['CUDA_VISIBLE_DEVICES']' in 'run.py' set to the selected gpus. The first card is the main card for calculation, which requires a little more memory.

  - Logs for training are in the `log` folder and the trained model is saved in the `checkpoints` folder.

  ```bash
  python run_bert.py or python run_crflstm.py
  ```

- Prediction
    
   Chinese datasets are supported by default. If English datasets are used, 'nltk' need to be installed and download the corresponding vocabulary by running 'nltk.download('punkt')'. **Meanwhile before prediction, 'lan' in *config.yaml* also need to be set *en*.**

  ```bash
  python predict.py
  ```

## Model

BiLSTM + CRF

BERT

## Prepare weak_supervised data

If you only have text data and corresponding dictionaries, but no canonical training data.

You can get weakly supervised formatted training data through automated labeling methods.

Please make sure that:

- Provide high-quality dictionaries
- Enough text data

<p align="left">
<a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data/README.md">prepare-data</a> </b>
</p>
