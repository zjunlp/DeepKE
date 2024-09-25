# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/triple/mt5/README_ZH.md">简体中文</a> </b>
</p>


## 1. Preparation

### Clone Code

```bash
git clone  https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/triple/mt5
```


### Create a Python virtual environment and install using pip

```bash
conda create -n ccks-mt5 python=3.9   
conda activate ccks-mt5
pip install -r requirements.txt
```


### Download data
Download  `train.json` and `valid.json`  (although the name is valid, this is not a validation set, but a test set for the competition) from official website https://tianchi.aliyun.com/competition/entrance/, and place them in the directory `./data`


## 2. Run

You can start the finetune model using the following script:

```bash
bash run_finetene_ds.bash
```

Alternatively, set your own parameters to execute using the following command:

```bash
deepspeed  --include localhost:0,1 run_finetune.py \
    --do_train --do_eval --do_predict \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 48 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --from_checkpoint=True \
    --overwrite_output_dir=False \
    --model_name_or_path google/mt5-base   \
    --output_dir output/ccks_mt5-base_f1_1e-4  \
    --logging_dir output/ccks_mt5-base_f1_1e-4_log \
    --train_file data/train.json \
    --test_file data/valid.json \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --metric_for_best_model "overall-score" \
    --learning_rate 1e-4 \
    --use_fast_tokenizer=True \
    --preprocessing_num_workers 4 \
    --generation_max_length 256 \
    --generation_num_beams 1 \
    --gradient_checkpointing=True \
    --deepspeed "configs/ds_mt5_z3_config_bf16.json" \
    --seed 42 \
    --bf16=True \
```

1.In DeepSpeed, `--include localhost:0,1` needs to be used instead of `CUDA_VISIBLE_DEFICES="0,1"`, the effect is the same
2. For different runs (different settings), you need to modify the parameter `output_dir`、`logging_dir`, otherwise the new run will overwrite the old run 
3. You can use `--validation_file` provides a validation set or does nothing (in `run_finetune.py`, we will divide 20% of the samples from train.json into validation sets)
4. For more information on parameters, please refer to `./arguments.py` and [Transformers:TrainingArgument](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/trainer#transformers.TrainingArguments). Please review the configuration of DeepSpeed https://www.deepspeed.ai/docs/config-json/

 
## 3.Inference or Predict
You can use the following command to Inference and predict the trained model, model_name is the path of the trained model, output_dir is the output path

```python
model_name="output/ccks_mt5-base_f1_1e-4"
output_dir="output/ccks_mt5-base_f1_1e-4_test_result"
data_dir="data"

deepspeed  --include localhost:0 run_finetune.py \
    --do_predict \
    --predict_with_generate \
    --use_fast_tokenizer=True \
    --per_device_eval_batch_size 16 \
    --test_file=${data_dir}/valid.json \
    --model_name_or_path=${model_name}   \
    --output_dir=${output_dir}  \
    --overwrite_output_dir=False \
    --logging_dir=${output_dir}_log \
    --preprocessing_num_workers 4 \
    --generation_max_length 256 \
    --generation_num_beams 1 \
    --gradient_checkpointing=True \
    --bf16=True \
    --deepspeed "configs/ds_mt5_z3_config_bf16.json" \
    --seed 42 
```



## 4. Format Conversion
The `bash run_finetene_ds.bash` command mentioned above will output a file named `test_preds.json` in the `output/ccks_mt5-base_f1_1e-4` directory. Each line of the file contains only the word 'output'. If you need to meet the submission format requirements of the CCKS2023 competition, you also need to extract 'kg' from 'output'. Here is a simple example script called `convert.py`.

```bash
python convert.py \
    --src_path="data/valid.json" \
    --pred_path="output/ccks_mt5-base_f1_1e-4/test_preds.json" \
    --tgt_path="output/valid_result.json" \
```


## 5. Hardware
We conducted finetune on the model on two pieces of 'RTX3090 24GB' If your device configuration is higher or lower, please choose a different scale model or adjust parameters `batch_size`, `gradient_accumulation_steps`.


## 6.Acknowledgment
Part of the code comes from [deepseed plan t5 summarization. ipynb](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/deepseed-flan-t5-summarization.ipynb) many thanks.

