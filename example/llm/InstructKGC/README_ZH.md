# 快速上手

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README.md">English</a> | 简体中文 </b>
</p>



## 1.准备

### 克隆代码
```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/llm/InstructKGC
```

### 创建python虚拟环境, 并使用pip安装
```bash
conda create -n instructkgc python=3.9   
conda activate instructkgc
pip install -r requirements.txt
```

### 下载数据
Download  from 从官网https://tianchi.aliyun.com/competition/entrance/532080/information 下载文件 `train.json` 和 `valid.json` (虽然名字是valid, 但这不是验证集, 而是比赛的测试集)并放在目录 `./data` 中.


## 2.运行

你可以通过下面的脚本使用LoRA方法来finetune模型:

```bash
bash scripts/run_finetene.bash
```

或者通过下面的命令设置自己的参数执行:

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

1. 你可以使用`--valid_file`提供验证集, 或者什么都不做(在`finetune.py`中, 我们会从train.json中划分`val_set_size`数`量的样本做验证集), 你也可以使用`val_set_size`调整验证集的数量
2. `batch_size`、`micro_train_batch_size`、`gradient_accumulation_steps`、`GPU数量`的关系是 gradient_accumulation_steps = batch_size // micro_batch_size // GPU数量。micro_train_batch_size才是在每块GPU上执行的真实batch_size。

我们也提供了多GPU版本的LoRA训练脚本:

```bash
bash scripts/run_finetene_mul.bash
```


## 4.预测

你可以通过下面的脚本使用训练好的LoRA模型在比赛测试集上预测输出:

```bash
bash scripts/run_inference.bash
```

或者通过下面的命令设置自己的参数执行:

```bash
CUDA_VISIBLE_DEVICES="0" python inference.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'lora/llama-7b-e3-r8' \
    --input_file 'data/valid.json' \
    --output_file 'result/output_llama_7b_e3_r8.json' \
    --load_8bit \
```


## 5.格式转换
上面的 `bash run_inference.bash` 会在 `result` 目录下输出 `output_llama_7b_e3_r8.json` 文件, 文件中不包含 'kg' 字段, 如果需要满足CCKS2023比赛的提交格式还需要从 'output' 中抽取出 'kg', 这里提供一个简单的样例 `convert.py`

```bash
python utils/convert.py \
    --pred_path "result/output_llama_7b_e3_r8.json" \
    --tgt_path "result/result_llama_7b_e3_r8.json" 
```


## 6.硬件
我们在1块 上对模型进行了finetune
注意：请确保你的设备或服务器有足够的RAM内存！！！


## 7.Acknowledgment

代码基本来自于[Alpaca-LoRA](https://github.com/tloen/alpaca-lora), 仅做了部分改动, 感谢！
