# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/multimodal/README_CN.md">简体中文</a> </b>
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
cd DeepKE/example/re/multimodal
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    The MRE dataset comes from [https://github.com/thecharm/Mega](https://github.com/thecharm/Mega), many thanks.

    You can download the MRE dataset with detected visual objects using folloing command:

    ```bash
    wget 120.27.214.45/Data/re/multimodal/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - The dataset [MRE](https://github.com/thecharm/Mega) with detected visual objects is stored in `data`:
    - `img_detect`：Detected objects using RCNN
    - `img_vg`：Detected objects using visual grouding

    - `img_org`： Original images

    - `txt`: Text set

    - `vg_data`：Bounding image and `img_vg`

    - `ours_rel2id.json` Relation set

- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.

  - Training on MRE

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

## Model


IFAformer is a novel dual Multimodal Transformer model
with implicit feature alignment for the RE task, which utilizes the Transformer structure uniformly in the visual and textual without explicitly designing modal alignment structure