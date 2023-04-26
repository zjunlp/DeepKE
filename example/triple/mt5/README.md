## 1.Install

```
conda create -n ccks-mt5 python=3.9 
conda activate ccks-mt5
pip install -r requirements.txt
```

## 2.Prepare

Put `train.json` and `valid.json` in `./data` dictionary

## 3. Run

Run script.

```
bash run_finetene_ds.bash
```

Or run `run_finetene.py`.

More detail information of arguments see `./arguments.py` and [transformers:TrainingArgument](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/trainer#transformers.TrainingArguments) ，deepspeed config see https://www.deepspeed.ai/docs/config-json/

```
deepspeed  --include localhost:2,3 run_finetune.py \
    --do_train --do_eval --do_predict \
    --predict_with_generate \
    --model_name_or_path google/mt5-bas   \
    --output_dir output/ccks_mt5-base_f1_1e-4  \
    --overwrite_output_dir=False \
    --logging_dir output/ccks_mt5-base_f1_1e-4_log \
    --train_file data/train.json \
    --test_file data/valid.json \
    --use_fast_tokenizer=True \
    --from_checkpoint=True \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --metric_for_best_model "overall-score" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 48 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 4 \
    --generation_max_length 256 \
    --generation_num_beams 1 \
    --gradient_checkpointing=True \
    --bf16=True \
    --deepspeed "data/configs/ds_mt5_z3_config_bf16.json" \
    --seed 42 \

```

## 4.Hardware

2 x RTX3090 24GB


## 5.Result

mT5-base: 54.32

## 6.Acknowledgment

Part of our code is borrowed from [deepseed-flan-t5-summarization.ipynb](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/deepseed-flan-t5-summarization.ipynb), many thanks.