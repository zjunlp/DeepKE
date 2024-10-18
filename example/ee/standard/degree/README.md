# Easy Start

<p align="left">
    <b> English | <a href="./README_CN.md">简体中文</a> </b>
</p>

## Requirements

```bash
cd ..
pip install -r requirements
```

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ee/standard/degree
```

## Dataset Processing

- Select the data split by modifying the parameters in `.conf/config.yaml`, the default setting is `001`.
- Follow the instruction [here](./data/ACE/README.md) to get the `ACE` dataset.
- Run the command `python generate_data.py` to get the corresponding processed data, the processed data of default setting will be put in folder `./processed_data/ace_001`.
  
## Train
- Parameters, model paths and configuration for training are in the `.conf/config.yaml` folder and users can modify them before training.
- Then run the following command,
```bash 
python run.py
```

## Predict
- Change the parameter `e2e_model` in `conf/config.yaml` for your trained model.
- Then run the following command,
```bash 
python predict.py
```