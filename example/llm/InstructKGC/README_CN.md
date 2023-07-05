# InstructionKGC-指令驱动的自适应知识图谱构建

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README.md">English</a> | 简体中文 </b>
</p>


- [InstructionKGC-指令驱动的自适应知识图谱构建](#instructionkgc-指令驱动的自适应知识图谱构建)
  - [1.任务目标](#1任务目标)
  - [2.数据](#2数据)
  - [3.准备](#3准备)
    - [环境](#环境)
    - [下载数据](#下载数据)
    - [模型](#模型)
  - [4.LLaMA系列](#4llama系列)
    - [LoRA微调LLaMA](#lora微调llama)
    - [LoRA微调智析](#lora微调智析)
    - [预测](#预测)
  - [5.ChatGLM](#5chatglm)
    - [LoRA微调ChatGLM](#lora微调chatglm)
    - [P-Tuning微调ChatGLM](#p-tuning微调chatglm)
    - [预测](#预测-1)
  - [6.CPM-Bee](#6cpm-bee)
    - [OpenDelta Fine-tuning with CPM-Bee](#OpenDelta微调cpm-bee)
  - [7.格式转换](#7格式转换)
  - [8.硬件](#8硬件)
  - [9.Acknowledgment](#9acknowledgment)
  - [Citation](#citation)


## 1.任务目标

根据用户输入的指令抽取相应类型的实体和关系，构建知识图谱。其中可能包含知识图谱补全任务，即任务需要模型在抽取实体关系三元组的同时对缺失三元组进行补全。

以下是一个**知识图谱构建任务**例子，输入一段文本`input`和`instruction`（包括想要抽取的实体类型和关系类型），以`(ent1,rel,ent2)`的形式输出`input`中包含的所有关系三元组`output`：

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```

知识图谱补齐的含义是，在输入`miss_input`（`input`中缺失了一段文字）和`instruction`的情况下，模型仍然能够补齐缺失的三元组，输出`output`。下面是一个例子：

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
miss_input="2006年，弗雷泽出战中国天津举行的女子水球世界杯。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"。
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```

虽然`miss_input`中不包含“协助国家队夺得冠军”这段文字，但是模型能够补齐缺失的三元组，即仍然需要输出`(弗雷泽,属于,国家队)(国家队,夺得,冠军)`。

## 2.数据

比赛数据的训练集每条数据包含如下字段：

|    字段     |                          说明                          |
| :---------: | :----------------------------------------------------: |
|     id      |                     样本唯一标识符                     |
|    input    |    模型输入文本（需要抽取其中涉及的所有关系三元组）    |
| instruction |                 模型进行抽取任务的指令                 |
| output      | 模型期望输出，以(ent1,relation,ent2)形式组成的输出文本 |
|     kg      |                  input中涉及的知识图谱                  |

在测试集中仅包含`id`、`instruction`、`input`三个字段。


## 3.准备
### 环境
请参考[DeepKE/example/llm/README_CN.md](../README_CN.md/#环境依赖)创建python虚拟环境, 然后激活该环境 `deepke-llm`:
```
conda activate deepke-llm
```


### 下载数据
```bash
mkdir result
mkdir lora
mkdir data
```

从官网https://tianchi.aliyun.com/competition/entrance/532080/information 下载文件 `train.json` 和 `valid.json` (虽然名字是valid, 但这不是验证集, 而是比赛的测试集)并放在目录 `./data` 中.


### 模型
下面是一些模型
* [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf)
* [LLaMA-13b](https://huggingface.co/decapoda-research/llama-13b-hf)
* [Alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)
* [zhixi-13b-diff](https://huggingface.co/zjunlp/zhixi-13b-diff)
* [fnlp/moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
* [openbmb/cpm-bee-5b](https://huggingface.co/openbmb/cpm-bee-5b)
* [Linly-AI/ChatFlow-7B](https://huggingface.co/Linly-AI/ChatFlow-7B)
* [Linly-AI/Chinese-LLaMA-7B](https://huggingface.co/Linly-AI/Chinese-LLaMA-7B)




## 4.LLaMA系列

### LoRA微调LLaMA

你可以通过下面的命令设置自己的参数使用LoRA方法来微调模型:

```bash
CUDA_VISIBLE_DEVICES="0" python finetune_llama.py \
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

我们也提供了多GPU版本的LoRA训练命令:

```bash
CUDA_VISIBLE_DEVICES="0,1,2" torchrun --nproc_per_node=3 --master_port=1331 finetune_llama.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --train_path 'data/train.json' \
    --output_dir 'lora/llama-7b-e3-r8' \
    --batch_size 960 \
    --micro_train_batch_size 10 \
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


### LoRA微调智析
请参考[KnowLM2.2预训练模型权重获取与恢复](https://github.com/zjunlp/KnowLM#2-2)获得完整的智析模型权重。

注意: 由于智析已经在大量的信息抽取指令数据集上经过LoRA训练, 因此可以跳过这一步直接执行预测, 你也可以选择进一步训练。

大致遵循上面的[LoRA微调LLaMA](./README_CN.md/#lora微调llama)命令, 仅需做出下列修改
```bash
--base_model 'path to ZhiXi'
--output_dir 'lora/cama-13b-e3-r8' \
```


### 预测
以下是训练好的一些LoRA版本:
* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [zhixi-13B-LoRA](https://huggingface.co/zjunlp/zhixi-13b-lora/tree/main)

你可以通过下面的命令设置自己的参数执行来使用训练好的LoRA模型在比赛测试集上预测输出:

```bash
CUDA_VISIBLE_DEVICES="0" python inference_llama.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'lora/llama-7b-e3-r8' \
    --input_file 'data/valid.json' \
    --output_file 'result/output_llama_7b_e3_r8.json' \
    --load_8bit \
```


## 5.ChatGLM

### LoRA微调ChatGLM
你可以通过下面的命令使用LoRA方法来finetune模型:

```bash
deepspeed --include localhost:0 finetuning_lora.py \
  --train_path data/train.json \
  --model_dir /model \
  --num_train_epochs 5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --output_dir output_dir_lora/ \
  --log_steps 10 \
  --max_len 400 \
  --max_src_len 450 \
  --lora_r 8
```

### P-Tuning微调ChatGLM
你可以通过下面的命令使用P-Tuning方法来finetune模型:


```bash
deepspeed --include localhost:0 finetuning_pt.py \
  --train_path data/train.json \
  --model_dir /model \
  --num_train_epochs 20 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --output_dir output_dir_pt \
  --log_steps 10 \
  --max_len 768 \
  --max_src_len 450 \
  --pre_seq_len 16 \
  --prefix_projection true
```

### 预测

你可以通过下面的命令使用训练好的LoRA模型在比赛测试集上预测输出:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_chatglm_lora.py \
  --test_path data/valid.json \
  --device 0 \
  --ori_model_dir /model \
  --model_dir /output_dir_lora/global_step- \
  --max_len 768 \
  --max_src_len 450
```

你可以通过下面的命令使用训练好的P-Tuning模型在比赛测试集上预测输出:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_chatglm_pt.py \
  --test_path data/valid.json \
  --device 0 \
  --ori_model_dir /model \
  --model_dir /output_dir_lora/global_step- \
  --max_len 768 \
  --max_src_len 450
```

## 6.CPM-Bee
### OpenDelta微调CPM-Bee
首先，你需要将比赛的的trans.json转换为cpm-bee所要求的格式，并提取20%的样本作为测试集使用。

```python
import json
import random

with open('train.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f]

num_data = len(data)
num_eval = int(num_data * 0.2)  # 20%作为验证集
eval_data = random.sample(data, num_eval)

with open('train.jsonl', 'w', encoding='utf-8') as f:
    for d in data:
        if d not in eval_data:
            cpm_d={"input":d["input"],"prompt":d["instruction"],"<ans>":d["output"]}
            json.dump(cpm_d, f, ensure_ascii=False)
            f.write('\n')

with open('eval.jsonl', 'w', encoding='utf-8') as f:
    for d in eval_data:
        cpm_d={"input":d["input"],"prompt":d["instruction"],"<ans>":d["output"]}
        json.dump(cpm_d, f, ensure_ascii=False)
        f.write('\n')
```

将处理好的数据放入bee_data/文件夹，并且[CPM-Bee](https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune)提供的数据处理方法将其转为二进制文件

```bash
python ../../src/preprocess_dataset.py --input bee_data --output_path bin_data --output_name ccpm_data
```

修改模型微调脚本scripts/finetune_cpm_bee.sh

```bash
#! /bin/bash
# 四卡微调
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12346

OPTS=""
OPTS+=" --use-delta"  # 使用增量微调（delta-tuning）
OPTS+=" --model-config config/cpm-bee-10b.json"  # 模型配置文件
OPTS+=" --dataset ../tutorials/basic_task_finetune/bin_data/train"  # 训练集路径
OPTS+=" --eval_dataset ../tutorials/basic_task_finetune/bin_data/eval"  # 验证集路径
OPTS+=" --epoch 5"  # 训练epoch数
OPTS+=" --batch-size 5"    # 数据批次大小
OPTS+=" --train-iters 100"  # 用于lr_schedular
OPTS+=" --save-name cpm_bee_finetune"  # 保存名称
OPTS+=" --max-length 2048" # 最大长度
OPTS+=" --save results/"  # 保存路径
OPTS+=" --lr 0.0001"    # 学习率
OPTS+=" --inspect-iters 100"  # 每100个step进行一次检查(bmtrain inspect)
OPTS+=" --warmup-iters 1". # 预热学习率的步数为1
OPTS+=" --eval-interval 50"  # 每50步验证一次
OPTS+=" --early-stop-patience 5"  # 如果验证集loss连续5次不降，停止微调
OPTS+=" --lr-decay-style noam"  # 选择noam方式调度学习率
OPTS+=" --weight-decay 0.01"  # 优化器权重衰减率为0.01
OPTS+=" --clip-grad 1.0"  # 半精度训练的grad clip
OPTS+=" --loss-scale 32768"  # 半精度训练的loss scale
OPTS+=" --start-step 0"  # 用于加载lr_schedular的中间状态
OPTS+=" --load ckpts/pytorch_model.bin"  # 模型参数文件

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} finetune_cpm_bee.py ${OPTS}"

echo ${CMD}
$CMD
```
运行脚本
```bash
cd ../../src
bash scripts/finetune_cpm_bee.sh
```
详见CPM[官方微调教程](https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune)


## 6.格式转换
上面的 `bash run_inference.bash` 会在 `result` 目录下输出 `output_llama_7b_e3_r8.json` 文件, 文件中不包含 'kg' 字段, 如果需要满足CCKS2023比赛的提交格式还需要从 'output' 中抽取出 'kg', 这里提供一个简单的样例 `convert.py`

```bash
python utils/convert.py \
    --pred_path "result/output_llama_7b_e3_r8.json" \
    --tgt_path "result/result_llama_7b_e3_r8.json" 
```


## 7.硬件
我们在1块 上对模型进行了finetune
注意：请确保你的设备或服务器有足够的RAM内存！！！


## 8.Acknowledgment

代码基本来自于[Alpaca-LoRA](https://github.com/tloen/alpaca-lora), 仅做了部分改动, 感谢！

## Citation

如果您使用了本项目代码或数据，烦请引用下列论文:
```bibtex
@article{DBLP:journals/corr/abs-2305-11527,
  author       = {Honghao Gui and
                  Jintian Zhang and
                  Hongbin Ye and
                  Ningyu Zhang},
  title        = {InstructIE: {A} Chinese Instruction-based Information Extraction Dataset},
  journal      = {CoRR},
  volume       = {abs/2305.11527},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.11527},
  doi          = {10.48550/arXiv.2305.11527},
  eprinttype    = {arXiv},
  eprint       = {2305.11527},
  timestamp    = {Thu, 25 May 2023 15:41:47 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-11527.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
