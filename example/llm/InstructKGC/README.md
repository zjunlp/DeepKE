# InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README_CN.md">简体中文</a> </b>
</p>

- [InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction](#instructkgc-ccks2023-evaluation-of-instruction-based-knowledge-graph-construction)
  - [1.Task Object](#1task-object)
  - [2.Data](#2data)
  - [3.Preparation](#3preparation)
    - [Environment](#environment)
    - [Download data](#download-data)
    - [Model](#model)
  - [4.LLaMA-series](#4llama-series)
    - [LoRA Fine-tuning with LLaMA](#lora-fine-tuning-with-llama)
    - [LoRA Fine-tuning with ZhiXi (智析)](#lora-fine-tuning-with-zhixi-智析)
    - [Prediction](#prediction)
  - [5.ChatGLM](#5chatglm)
    - [Lora Fine-tuning with ChatGLM](#lora-fine-tuning-with-chatglm)
    - [P-Tuning Fine-tuning with ChatGLM](#p-tuning-fine-tuning-with-chatglm)
    - [Prediction](#prediction-1)
  - [6.CPM-Bee](#6cpm-bee)
    - [OpenDelta fine-tuning with CPM-Bee](#opendelta-fine-tuning-with-cpm-bee)
  - [7.Format Conversion](#7format-conversion)
  - [8.Hardware](#8hardware)
  - [9.Acknowledgment](#9acknowledgment)
  - [Citation](#citation)


## 1.Task Object

Extract relevant entities and relations according to user input instructions to construct a knowledge graph. This task may include knowledge graph completion, where the model is required to complete missing triples while extracting entity-relation triples.

Below is an example of a **Knowledge Graph Construction Task**. Given an input text `input` and an `instruction` (including the desired entity types and relationship types), output all relationship triples `output` in the form of `(ent1, rel, ent2)` found within the `input`:

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```

The meaning of knowledge graph completion is that, when given an input `miss_input` (a portion of the text is missing) and an `instruction`, the model is still able to complete the missing triples and output `output`. Here is an example:

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
miss_input="2006年，弗雷泽出战中国天津举行的女子水球世界杯。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"。
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```

Although the text "协助国家队夺得冠军" is not included in `miss_input`, the model can still complete the missing triples, i.e., it still needs to output `(弗雷泽,属于,国家队)(国家队,夺得,冠军)`.



## 2.Data

The training dataset for the competition contains the following fields for each data entry:

|    Field    |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|     id      |                   Sample unique identifier                   |
|    input    | Model input text (need to extract all triples involved within) |
| instruction |   Instruction for the model to perform the extraction task   |
|    output   | Expected model output, in the form of output text composed of (ent1, relation, ent2) |
|     kg      |             Knowledge graph involved in the input             |

In the test set, only the three fields `id`, `instruction`, and `input` are included.




## 3.Preparation

### Environment
Please refer to [DeepKE/example/llm/README.md](../README.md/#requirements) to create a Python virtual environment, and activate the `deepke-llm` environment:
```
conda activate deepke-llm
```

### Download data

```bash
mkdir result
mkdir lora
mkdir data
```

Download  `train.json` and `valid.json`  (although the name is valid, this is not a validation set, but a test set for the competition) from the official website https://tianchi.aliyun.com/competition/entrance/532080/information, and place them in the directory `./data`


### Model 
Here are some models:
* [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf)
* [Alpaca-7b](https://huggingface.co/circulus/alpaca-7b)
* [LLaMA-13b](https://huggingface.co/decapoda-research/llama-13b-hf)
* [Alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)
* [zhixi-13b-diff](https://huggingface.co/zjunlp/zhixi-13b-diff)
* [fnlp/moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
* [openbmb/cpm-bee-5b](https://huggingface.co/openbmb/cpm-bee-5b)
* [Linly-AI/ChatFlow-7B](https://huggingface.co/Linly-AI/ChatFlow-7B)
* [Linly-AI/Chinese-LLaMA-7B](https://huggingface.co/Linly-AI/Chinese-LLaMA-7B)



## 4.LLaMA-series

### LoRA Fine-tuning with LLaMA

You can use the LoRA method to fine-tune the model by setting your own parameters using the following command:

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


We also provide multiple GPU versions of LoRA training commands:

```bash
CUDA_VISIBLE_DEVICES="0,1,2" torchrun --nproc_per_node=3 --master_port=1331 finetune.py \
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


### LoRA Fine-tuning with ZhiXi (智析)
Please refer to [KnowLM2.2Pre-trained Model Weight Acquisition and Restoration](https://github.com/zjunlp/KnowLM#2-2) to obtain the complete ZhiXi model weights.

Note: Since ZhiXi has already been trained with LoRA on a large-scale information extraction instruction dataset, you can skip this step and proceed directly to Step 3 Prediction. If you wish to refine the model further, additional training remains an option.

Follow the command mentioned above [LoRA Fine-tuning with LLaMA](./README.md/#lora-fine-tuning-with-llama) with the following modifications.
```bash
--base_model 'path to zhixi'
--output_dir 'lora/cama-13b-e3-r8' 
```

### Prediction
Here are some trained versions of LoRA:
* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [zhixi-13B-LoRA](https://huggingface.co/zjunlp/zhixi-13b-lora/tree/main)

You can use the following command to set your own parameters and execute it to make predictions using the trained LoRA model on the competition test dataset:

```bash
CUDA_VISIBLE_DEVICES="0" python inference.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'lora/llama-7b-e3-r8' \
    --input_file 'data/valid.json' \
    --output_file 'result/output_llama_7b_e3_r8.json' \
    --load_8bit \
```



## 5.ChatGLM


### Lora Fine-tuning with ChatGLM
You can use the LoRA method to finetune the model using the following script:

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

### P-Tuning Fine-tuning with ChatGLM
You can use the P-Tuning method to finetune the model using the following script:

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

### Prediction
You can use the trained LoRA model to predict the output on the competition test set using the following script:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_chatglm_lora.py \
  --test_path data/valid.json \
  --device 0 \
  --ori_model_dir /model \
  --model_dir /output_dir_lora/global_step- \
  --max_len 768 \
  --max_src_len 450
```
You can use the trained P-Tuning model to predict the output on the competition test set using the following script:

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
### OpenDelta fine-tuning with CPM-Bee
First you need to convert the event's trans.json into the format required by cpm-bee and extract 20% of the sample to be used as testers

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

put it in bee_data/ ,and use the data process tool privided by [CPM-Bee](https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune)

```bash
python ../../src/preprocess_dataset.py --input bee_data --output_path bin_data --output_name ccpm_data
```

Modify the model fine-tuning script scripts scripts/finetune_cpm_bee.sh to

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
run the bash
```bash
cd ../../src
bash scripts/finetune_cpm_bee.sh
```
For more details check out the [official tutoris](https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune)


## 7.Format Conversion
The `bash run_inference.bash` command mentioned above will output a file named `output_llama_7b_e3_r8.json` in the `result` directory, which does not contain the 'kg' field. If you need to meet the submission format requirements of the CCKS2023 competition, you also need to extract 'kg' from 'output'. Here is a simple example script called `convert.py`.


```bash
python utils/convert.py \
    --pred_path "result/output_llama_7b_e3_r8.json" \
    --tgt_path "result/result_llama_7b_e3_r8.json" 
```


## 8.Hardware
We performed finetune on the model on 1 `RTX3090 24GB`
Attention: Please ensure that your device or server has sufficient RAM memory!!!


## 9.Acknowledgment
The code basically comes from [Alpaca-LoRA](https://github.com/tloen/alpaca-lora). Only some changes have been made, many thanks.

## Citation

If you use this project, please cite the following papers:
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
