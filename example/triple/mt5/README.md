## 1.Install

```
conda create -n ccks-mt5 python=3.9 
conda activate ccks-mt5
pip install -r requirements.txt
```

## 2.Prepare
Download train.json and valid.json from https://tianchi.aliyun.com/competition/entrance/532080/information and place them in the `./data` directory.


## 3. Run

You can perform finetune through scripts:

```
bash run_finetene_ds.bash
```

Alternatively, use the following command to set your own parameters for execution:

```
deepspeed  --include localhost:0,1 run_finetune.py \
    --do_train --do_eval --do_predict \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 48 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 96 \
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


I. In DeepSpeed, `--include localhost:0,1` are suggested to replace `CUDA_VISIBLE_DEVICES="0,1"`.
II. You need to change `output_dir`、`logging_dir` for different runs. Otherwise, the new run will overwrite the old run.

For more information on parameters, please refer to `./arguments.py` and [Transformers:TrainingArgument](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/trainer#transformers.TrainingArguments)
DeepSpeed config see https://www.deepspeed.ai/docs/config-json/


## 4.Hardware
Our device is `2 x RTX3090 24GB`. If you are not, please select a model of a different scale or adjust the parameters of `batch_size`, `gradient_accumulation_steps`, `eval_accumulation_steps`.


## 5.Result
After training for 10 epochs, The results are as follows
| model    | Score |
| -------- | ----- |
| mt5-base | 54.32 |


## 6.Acknowledgment

Part of our code is borrowed from [deepseed-flan-t5-summarization.ipynb](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/deepseed-flan-t5-summarization.ipynb), many thanks.