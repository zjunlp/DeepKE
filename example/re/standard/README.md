# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/standard/README_CN.md">简体中文</a> </b>
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
- deepke 

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/standard
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    ```bash
    wget 120.27.214.45/Data/re/standard/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - Three types of data formats are supported,including `json`,`xlsx` and `csv`. The dataset is stored in `data/origin`:
    - `train.csv`: Training set
    -  `valid.csv `: Validation set
    - `test.csv`: Test set
    - `relation.csv`: Relation labels
  
- Training

  - Parameters for training are in the `conf` folder and users can modify them before training.
  - If using LM, modify 'lm_file' to use the local model.
  - Logs for training are in the `log` folder and the trained model is saved in the `checkpoints` folder. This task supports multi card training. Modify `trian.yaml`'s parameter `use_multi_gpu` to true, `gpu_ids` set to the selected gpus. The first card is the main card for calculation, which requires a little more memory.`show_plot` set to visualize the loss of the current epoch.The default value is False.



  ```bash
  python run.py
  ```

- Prediction

    Set the *fp* in [predict.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/re/standard/conf/predict.yaml) as the path of the trained model / checkpoint to be used in prediction.The absolute path of the model needs to be used，such as `xxx/checkpoints/2019-12-03_ 17-35-30/cnn_ epoch21.pth`.
  ```bash
  python predict.py
  ```

## Models

1. CNN
2. RNN
3. Capsule
4. GCN (Based on the paper ["Graph Convolution over Pruned Dependency Trees Improves Relation Extraction"](https://aclanthology.org/D18-1244.pdf))
5. Transformer
6. Pre-trained Model (BERT)

## Data Labeling

If you only have sentence and entity pairs but relation labels, you can get use our distant supervised based [relation labeling tools](https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data).

Please make sure that:

- Use the triple file we provide or high-quality customized triple file
- Enough source data 
