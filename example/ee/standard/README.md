# Easy Start

<p align="left">
    <b> English | <a href="./README_CN.md">简体中文</a> </b>
</p>

## Requirements

To install requirements:

```
python==3.8
Java 8
```

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ee/few-shot
```

## Install with pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset
  - `ace05e`
  1. Prepare data processed from [DyGIE++](https://github.com/dwadden/dygiepp#ace05-event)
  2. Put the processed data into the folder `data/raw_data/ace05e_dygieppformat`

  - `ace05ep`
  1. Download ACE data from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06)
  2. Put the processed data into the folder `data/raw_data/ace_2005`

  - `preprocess`
  ```Bash
  cd data/
  bash ./scripts/process_ace05e.sh
  bash ./scripts/process_ace05ep.sh
  ```
  You can change the low resource split by modify the parameters in file `conf/generate_data.yaml`, and then run the following commands.
  ```Bahs
  cd ..
  python generate_data.py
  ```
  the input data will be generated and stored in new dictionary like `proceesed_data/degree_e2e_ace05ep_001`.


- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.
  ```bash
  python run.py
  ```

- Prediction

  - change the parameter `e2e_model` in `conf/predict.yaml` for your trained model.
  ```bash
  python predict.py
  ```