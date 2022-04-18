# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/multimodal/README_CN.md">简体中文</a> </b>
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
cd DeepKE/example/ner/multimodal
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset - Twitter2015 & Twitter2017

  - Download the dataset to this directory.

    The text data follows the conll format.
    The acquisition of Twitter15 and Twitter17 data refer to the code from [UMT](https://github.com/jefferyYu/UMT/), many thanks.

    You can download the Twitter2015 and Twitter2017 dataset with detected visual objects using folloing command:

    ```bash
    wget 120.27.214.45/Data/ner/multimodal/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - The twitter15 dataset with detected visual objects is stored in `data`:
    - `twitter15_detect`：Detected objects using RCNN
    - `twitter2015_aux_images`：Detected objects using visual grouding

    - `twitter2015_images`： Original images

    - `train.txt`: Train set

    - `...`

- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.

  - Run

    ```bash
    python run.py
    ```

  - The trained model is stored in the `checkpoint` directory by default and you can change it by modifying "save_path" in `train.yaml`.

  - Start to train from last-trained model<br>

    modify `load_path` in `train.yaml` as the path of the last-trained model

  - Logs for training are stored in the current directory by default and the path can be configured by modifying `log_dir` in `.yaml`

- Prediction
  
  Modify "load_path" in `predict.yaml` to the trained model path.
  ```bash
  python predict.py
  ```