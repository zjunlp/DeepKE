# Easy Start

<p align="left">
    <b> English | <a href="./README_CN.md">简体中文</a> </b>
</p>

## Requirements

- Create and enter the python virtual environment.

- To install requirements:
  ```bash
  python==3.8
  pip install -r requirements.txt
  ```

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ee/standard
```

## Dataset
- `ACE`
Follow the instruction [here](./data/ACE/README.md)

- `DuEE`
Follow the instruction [here](./data/DuEE/README.md)

## Train

Modify the parameters in `./conf/train.yaml`.

- Trigger
  Set `task_name` to `trigger`.
  Select different dataset by set `data_name`.
  Then run the following command:
  ```bash
  python run.py
  ```

- Role
  Here we train the event arguments extraction model with the gold trigger.
  Set `task_name` to `role`.
  Select different dataset by set `data_name`.
  Then run the following command:
  ```bash
  python run.py
  ```

## Predict

The trigger prediction has been conducted during training, and the result is in the `output_dir`.Here we predict the event arguments extraction results with pred trigger result.
Modify the parameters in `./conf/predict.yaml`.
Then run the following command:
```bash
  python predict.py
```