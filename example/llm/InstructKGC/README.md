# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README_ZH.md">简体中文</a> </b>
</p>


## 1. Preparation

### Clone Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/llm/InstructKGC
```


### Create a Python virtual environment and install using pip

```bash
conda create -n instructkgc python=3.9   
conda activate instructkgc
pip install -r requirements.txt
```


### Download data
Download  `train.json` and `valid.json`  (although the name is valid, this is not a validation set, but a test set for the competition) from official website https://tianchi.aliyun.com/competition/entrance/, and place them in the directory `./data`


## 2. Run

You can use the LoRA method to finetune the model using the following script:

```bash
bash scripts/run_finetene.bash
```

Alternatively, set your own parameters to execute using the following command:

```bash
CUDA_VISIBLE_DEVICES="0" python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --train_path 'data/train.json' \
    --output_dir 'lora/llama-7b-e3-r8' \
    --batch_size 128 \
    --micro_train_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
```

1. You can use `--valid_file` provides a validation set, or does nothing at all (in `finetune.py`, we will divide the number of samples with `val_set_size` from train.json as the validation set), you can also use `val_set_size` adjust the number of validation sets
2. `gradient_accumulation_steps` = `batch_size` // `micro_batch_size` // Number of GPU


We also provide multiple GPU versions of LoRA training scripts:

```bash
bash scripts/run_finetene_mul.bash
```



## 4. Predict or Inference
You can use the trained LoRA model to predict the output on the competition test set using the following script:

```bash
bash scripts/run_inference.bash
```

Alternatively, set your own parameters to execute using the following command:

```bash
CUDA_VISIBLE_DEVICES="0" python inference.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'lora/llama-7b-e3-r8' \
    --input_file 'data/valid.json' \
    --output_file 'result/output_llama_7b_e3_r8.json' \
    --load_8bit \
```


## 5. Format Conversion
The `bash run_inference.bash` command mentioned above will output a file named `output_llama_7b_e3_r8.json` in the `result` directory, which does not contain the 'kg' field. If you need to meet the submission format requirements of the CCKS2023 competition, you also need to extract 'kg' from 'output'. Here is a simple example script called `convert.py`.


```bash
python utils/convert.py \
    --pred_path "result/output_llama_7b_e3_r8.json" \
    --tgt_path "result/result_llama_7b_e3_r8.json" 
```


## 6. Hardware
We performed finetune on the model on 1 `RTX3090 24GB`
Attention: Please ensure that your device or server has sufficient RAM memory!!!


## 7.Acknowledgment
The code basically comes from [Alpaca-LoRA](https://github.com/tloen/alpaca-lora). Only some changes have been made, many thanks.