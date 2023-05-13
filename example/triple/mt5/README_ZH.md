# 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/tree/main/example/triple/mt5/README.md">English</a> | 简体中文 </b>
</p>



## 1.准备

### 克隆代码
```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/triple/mt5
```

### 创建python虚拟环境, 并使用pip安装
```bash
conda create -n ccks-mt5 python=3.9   
conda activate ccks-mt5
pip install -r requirements.txt
```

### 下载数据
Download  from 从官网https://tianchi.aliyun.com/competition/entrance/532080/information 下载文件 `train.json` 和 `valid.json` 并放在目录 `./data` 中.


## 2.运行

你可以通过下面的脚本开始finetune模型:

```bash
bash run_finetene_ds.bash
```


或者通过下面的命令设置自己的参数执行:

```bash
deepspeed  --include localhost:0,1 run_finetune.py \
    --do_train --do_eval \
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

1. 在DeepSpeed中, 需要使用`--include localhost:0,1` 来代替`CUDA_VISIBLE_DEVICES="0,1"`, 作用是一样的.
2. 对于不同的runs(不同的设定)你需要修改参数 `output_dir`、`logging_dir` , 否则新的run会覆盖旧的run.


更多有关参数的信息请查看`./arguments.py` 和 [Transformers:TrainingArgument](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/trainer#transformers.TrainingArguments)
有关DeepSpeed的配置请查看 https://www.deepspeed.ai/docs/config-json/


## 3.硬件
我们在2块 `RTX3090 24GB`上对模型进行了finetune. 如果你的设备配置更高或更低, 请选择其他规模的模型或调整参数 `batch_size`, `gradient_accumulation_steps`.


## 4.Acknowledgment

部分代码来自于 [deepseed-flan-t5-summarization.ipynb](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/deepseed-flan-t5-summarization.ipynb), 感谢！
