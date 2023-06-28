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
  pip install hydra-core==1.3.1 # ignore the conlict with deepke
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

Modify the parameters in `./conf/train.yaml`. Select different dataset by set `data_name`, and change the `model_name_or_path` for different dataset.

- Trigger (Event Detection or Trigger Classification)
  
  First train trigger classification model, and predict the trigger of each instance.
  
  Set `task_name` to `trigger`.
  Then run the following command:
  
  ```bash
  python run.py
  ```
  
  The prediction will be conducted after the training, and the result will be in `exp/xx/trigger/xxx/eval_pred.json`.

Then train the event arguments extraction model by the gold trigger.
- Role (Event Arguments Extraction)
  Then, we train the event arguements extraction models, here we train the event arguments extraction model with the gold trigger.
  Set `task_name` to `role`.
  Then run the following command:
  
  ```bash
  python run.py
  ```

## Predict (Event Arguments Extraction)

The trigger prediction has been conducted during training, and the result is in the `output_dir`. Here we predict the event arguments extraction results with pred trigger result.

Modify the parameters in `./conf/predict.yaml`. Set the `model_name_or_path` to the trained role model path, and `do_pipeline_predict=True` to do the pipeline prediction.

Then run the following command:
```bash
  python predict.py
```
The final result will be in `eval_pred.json` of the role model path folder.