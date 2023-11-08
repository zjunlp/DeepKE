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
  - [4.LoRA Fine-tuning](#4lora-fine-tuning)
    - [LoRA Fine-tuning with LLaMA](#lora-fine-tuning-with-llama)
    - [LoRA Fine-tuning with Alpaca](#lora-fine-tuning-with-alpaca)
    - [LoRA Fine-tuning with ZhiXi (智析)](#lora-fine-tuning-with-zhixi-智析)
    - [Lora Fine-tuning with ChatGLM](#lora-fine-tuning-with-chatglm)
    - [Lora Fine-tuning with Moss](#lora-fine-tuning-with-moss)
    - [LoRA Fine-tuning with Baichuan](#lora-fine-tuning-with-baichuan)
  - [5.P-Tuning Fine-tuning](#5p-tuning-fine-tuning)
    - [P-Tuning Fine-tuning with ChatGLM](#p-tuning-fine-tuning-with-chatglm)
    - [Prediction of Lora](#prediction-of-lora)
    - [Prediction of P-Tuning](#prediction-of-p-tuning)
  - [6. Model Output Conversion \& F1 Calculation](#6-model-output-conversion--f1-calculation)
  - [7.Acknowledgment](#7acknowledgment)
  - [Citation](#citation)


## 1.Task Object

Extract relevant entities and relations according to user input instructions to construct a knowledge graph. This task may include knowledge graph completion, where the model is required to complete missing triples while extracting entity-relation triples.

Below is an example of a **Knowledge Graph Construction Task**. Given an input text `input` and an `instruction` (including the desired entity types and relationship types), output all relationship triples `output` in the form of `(ent1, rel, ent2)` found within the `input`:

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```



## 2.Data

The model's input should include the `instruction` and  the `input`(optionally) field. We have provided a script, [kg2instruction/convert.py](./kg2instruction/convert.py)、[kg2instruction/convert_test.py](./kg2instruction/convert_test.py), which is used to transform the data into a format suitable for direct input into the model.

* Note! Before executing [kg2instruction/convert.py](./kg2instruction/convert.py), please refer to the [data](./data) directory for the expected data format for each task.

```bash
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \      # Different templates and conversion scripts are used for different languages
  --task NER \         # 5 types of tasks: ['RE', 'NER', 'EE', 'EET', 'EEA']
  --sample -1 \        # If -1, randomly sample one from 20 instruction types and 4 output formats otherwise it is the specified instruction format, -1<=sample<20
  --neg_ratio 1 \      # Indicates the negative sampling ratio for all samples
  --neg_schema 1 \     # Indicates the negative sampling ratio from the schema
  --random_sort        # Whether to randomly sort the schema list in the instructions

```

[kg2instruction/convert_test.py](./kg2instruction/convert_test.py) does not require data to have label (`entity`, `relation`, `event`) fields, only needs to have an `input` field and provide a `schema_path` is suitable for processing test data.

```bash
python kg2instruction/convert_test.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/processed.json \
    --schema_path data/NER/schema.json \
    --language zh \      
    --task NER \          
    --sample 0 
```



* For more detailed information about IE templates, data format conversion, and data extraction, please refer to [kg2instruction/README.md](./kg2instruction/README.md).

* Alternatively, you can independently create data that includes the `instruction` and `input` fields.

Here are some readily processed datasets:

| Name                   | Download                                                     | Quantity | Description                                                  |
| ---------------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| InstructIE-train          | [Google drive](https://drive.google.com/file/d/1VX5buWC9qVeVuudh_mhc_nC7IPPpGchQ/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/1xXVrjkinw4cyKKFBR8BwQw?pwd=x4s7)  | 30w+  | InstructIE train set |
| InstructIE-valid       | [Google drive](https://drive.google.com/file/d/1EMvqYnnniKCGEYMLoENE1VD6DrcQ1Hhj/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/11u_f_JT30W6B5xmUPC3enw?pwd=71ie)     | 2000+ | InstructIE validation set                                                                                     |
| InstructIE-test       | [Google drive](https://drive.google.com/file/d/1WdG6_ouS-dBjWUXLuROx03hP-1_QY5n4/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)  <br/> [Baidu Netdisk](https://pan.baidu.com/s/1JiRiOoyBVOold58zY482TA?pwd=cyr9)     | 2000+ | InstructIE test set                                                                                    |
| train.json, valid.json | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing) | 5,000    | Preliminary training set and test set for the task "Instruction-Driven Adaptive Knowledge Graph Construction" in [CCKS2023 Open Knowledge Graph Challenge](https://tianchi.aliyun.com/competition/entrance/532080/introduction), randomly selected from instruct_train.json |


`InstrumentIE-train` contains two files: `InstrumentIE-zh.json` and `InstrumentIE-en.json`, each of which contains the following fields: `'id'` (unique identifier), `'cate'` (text category), `'entity'` and `'relation'` (triples) fields. The extracted instructions and output can be freely constructed through `'entity'` and `'relation'`.

`InstrumentIE-valid` and `InstrumentIE-test` are validation sets and test sets, respectively, including bilingual `zh` and `en`.

`train.json`: Same fields as `KnowLM-IE.json`, `'instruction'` and `'output'` have only one format, and extraction instructions and outputs can also be freely constructed through `'relation'`.

`valid.json`: Same fields as `train.json`, but with more accurate annotations achieved through crowdsour

Here is an explanation of each field:

|    Field    |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|     id      |                   Unique identifier                   |
|    cate     |     text topic of input (12 topics in total)                        |
|    input    | Model input text (need to extract all triples involved within) |
| instruction |   Instruction for the model to perform the extraction task   |
|    output   | Expected model output |
| entity      |            entities(entity, entity_type)                    |
|   relation  |             Relation triples(head, relation, tail) involved in the input             |


Here [schema](./kg2instruction/convert/utils.py) provides 12 text topics and common relationship types under the topic.


## 3.Preparation

### Environment
Please refer to [DeepKE/example/llm/README.md](../README.md/#requirements) to create a Python virtual environment, and activate the `deepke-llm` environment:

```bash
conda activate deepke-llm
```

!!! Attention: To accommodate the `qlora` technique, we have upgraded the versions of the `transformers`, `accelerate`, `bitsandbytes`, and `peft` libraries in the original deepke-llm codebase.

1. transformers 0.17.1 -> 4.30.2
2. accelerate 4.28.1 -> 0.20.3
3. bitsandbytes 0.37.2 -> 0.39.1
4. peft 0.2.0 -> 0.4.0dev


### Download data

```bash
mkdir results
mkdir lora
mkdir data
```

Place the data in the directory `./data`


### Model 
Here are some models:
* [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | [LLaMA-13b](https://huggingface.co/decapoda-research/llama-13b-hf)
* [Alpaca-7b](https://huggingface.co/circulus/alpaca-7b) | [Alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)
* [Vicuna-7b-delta-v1.1](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) | [Vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1)
* [zjunlp/knowlm-13b-base-v1.0](https://huggingface.co/zjunlp/knowlm-13b-base-v1.0) (requires corresponding IE Lora) | [zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi) (can predict directly without Lora) | [zjunlp/knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie) (stronger IE capabilities, but with reduced generalization, no need for Lora)
* [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
* [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
* [Chinese-LLaMA-7B](https://huggingface.co/Linly-AI/Chinese-LLaMA-7B)
* [baichuan-inc/Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B) | [baichuan-inc/Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) | [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) | [baichuan-inc/Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)



## 4.LoRA Fine-tuning

Basic Parameters:
* `--model_name`: The current code supports the following models ["llama", "falcon", "baichuan", "chatglm", "moss", "alpaca", "vicuna", "zhixi"], note the distinction from `model_name_or_path`.
* `--train_file`, `--valid_file` (optional): Training and validation set file paths (JSON format files), if `valid_file` is not specified, we will default to partitioning a `val_set_size` number of samples from the `train_file` to use as a validation set. You can also use `val_set_size` to adjust the number of samples in the validation set.
* `--output_dir`: Path for saving Lora weight parameters.
* `--val_set_size`: Number of samples in the validation set, default is 1000.
* `--prompt_template_name`: Template name, currently supports three types of templates [alpaca, vicuna, moss], with the default being the alpaca template.


> Attention!! All the following commands should be executed in the `InstrctKGC` directory!! For example, running a fine-tuning script would be: `bash scripts/fine_llama.bash`.

### LoRA Fine-tuning with LLaMA

You can use the following command to configure your own parameters and fine-tune the Llama model using the LoRA method:

```bash
output_dir='path to save Llama Lora'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --master_port=1331 src/finetune.py \
    --do_train --do_eval \
    --model_name_or_path 'path or name to Llama' \
    --model_name 'llama' \
    --train_file 'data/train.json' \
    --output_dir=${output_dir}  \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --optim "adamw_torch" \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_memory_MB 24000 \
    --fp16 \
    --bits 8 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. We use the [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf) model for Llama.
2. You can provide a validation set using the `--valid_file` option, or you can do nothing (in `finetune.py`, we will split a specified number of samples from `train.json` as the validation set using `val_set_size`). You can also adjust the number of samples in the validation set using `val_set_size`.
3. We use the default `alpaca` template for the `prompt_template_name`. Please refer to `templates/alpaca.json` for more details.
4. For more detailed parameter information, please refer to `utils/args.py`.
5. `max_memory_MB` (default 80000) specifies the GPU memory size. You need to specify it according to your own GPU capacity.
6. We have successfully run the Llama-LoRA fine-tuning code on an `RTX3090` GPU.
7. model_name = llama(The same for llama2)

The corresponding script can be found at `ft_scripts/fine_llama.bash`.




### LoRA Fine-tuning with Alpaca

To fine-tune Alpaca using the LoRA method, you can make the following modifications to the `scripts/fine_llama.bash` script, following the instructions outlined in [LoRA Fine-tuning with LLaMA](./README.md/#lora-fine-tuning-with-llama):


```bash
output_dir='path to save Alpaca Lora'
--model_name_or_path 'path or name to Alpaca' \
--model_name 'alpaca' \
```

1. We use the [Alpaca-7b](https://huggingface.co/circulus/alpaca-7b) model for Alpaca.
2. We use the default `alpaca` template for the `prompt_template_name`. Please refer to `templates/alpaca.json` for more details.
3. We have successfully run the Alpaca-LoRA fine-tuning code on an `RTX3090` GPU.
4. model_name = alpaca




### LoRA Fine-tuning with ZhiXi (智析)
Please refer to [KnowLM2.2Pre-trained Model Weight Acquisition and Restoration](https://github.com/zjunlp/KnowLM#2-2) to obtain the complete ZhiXi model weights.

Note: Since ZhiXi has already been trained with LoRA on a large-scale information extraction instruction dataset, you can skip this step and proceed directly to Step 3 Prediction. If you wish to refine the model further, additional training remains an option.


To fine-tune ZhiXi using the LoRA method, you can make the following modifications to the `scripts/fine_llama.bash` script, following the instructions outlined in [LoRA Fine-tuning with LLaMA](./README.md/#lora-fine-tuning-with-llama):


```bash
output_dir='path to save Zhixi Lora'
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--model_name_or_path 'path or name to Zhixi' \
--model_name 'zhixi' \
```

1. Since Zhixi currently only has a 13b model, you will need to decrease the batch size accordingly to accommodate the model's memory requirements.
2. We will continue to use the default `alpaca` template for the `prompt_template_name`. Please refer to `templates/alpaca.json` for more details.
3. We have successfully run the ZhiXi-LoRA fine-tuning code on an `RTX3090` GPU.
4. model_name = zhixi




### Lora Fine-tuning with ChatGLM

You can use the following command to configure your own parameters and fine-tune the ChatGLM model using the LoRA method:

```bash
output_dir='path to save ChatGLM Lora'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1" python --nproc_per_node=2 --master_port=1331 src/finetune.py \
    --do_train --do_eval \
    --model_name_or_path 'path or name to ChatGLM' \
    --model_name 'chatglm' \
    --train_file 'data/train.json' \
    --output_dir=${output_dir}  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --weight_decay 5e-4 \
    --adam_beta2 0.95 \
    --optim "adamw_torch" \
    --max_source_length 512 \
    --max_target_length 256 \
    --val_set_size 1000 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --max_memory_MB 24000 \
    --fp16 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. We use the [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) model for ChatGLM.
2. We use the default `alpaca` template for the `prompt_template_name`. Please refer to `templates/alpaca.json` for more details.
3. Due to unsatisfactory performance with 8-bit quantization, we did not apply quantization to the ChatGLM model.
4. `max_memory_MB` (default 80000) specifies the GPU memory size. You need to specify it according to your own GPU capacity.
5. We have successfully run the ChatGLM-LoRA fine-tuning code on an `RTX3090` GPU.
6. model_name = chatglm

The corresponding script can be found at `ft_scripts/fine_chatglm.bash`.




### Lora Fine-tuning with Moss

You can use the following command to configure your own parameters and fine-tune the Moss model using the LoRA method:

```bash
output_dir='path to save Moss Lora'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --master_port=1331 src/finetune.py \
    --do_train --do_eval \
    --model_name_or_path 'path or name to Moss' \
    --model_name 'moss' \
    --prompt_template_name 'moss' \
    --train_file 'data/train.json' \
    --output_dir=${output_dir}  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --num_train_epochs 10 \
    --learning_rate 2e-4 \
    --optim "paged_adamw_32bit" \
    --max_grad_norm 0.3 \
    --lr_scheduler_type 'constant' \
    --max_source_length 512 \
    --max_target_length 256 \
    --val_set_size 1000 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 16 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --max_memory_MB 24000 \
    --fp16 \
    --bits 4 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. We use the [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft) model for Moss.
2. The `prompt_template_name` has been modified based on the alpaca template. Please refer to `templates/moss.json` for more details. Therefore, you need to set `--prompt_template_name 'moss'`.
3. Due to memory limitations on the `RTX3090`, we use the `qlora` technique for 4-bit quantization. However, you can try 8-bit quantization or non-quantization strategies on `V100` or `A100` GPUs.
4. `max_memory_MB` (default 80000) specifies the GPU memory size. You need to specify it according to your own GPU capacity.
5. We have successfully run the Moss-LoRA fine-tuning code on an `RTX3090` GPU.
6. model_name = moss

The corresponding script can be found at `ft_scripts/fine_moss.bash`.



### LoRA Fine-tuning with Baichuan

You can use the following command to configure your own parameters and fine-tune the Llama model using the LoRA method:

```bash
output_dir='path to save Llama Lora'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --master_port=1331 src/finetune.py \
    --do_train --do_eval \
    --model_name_or_path 'path or name to Llama' \
    --model_name 'llama' \
    --train_file 'data/train.json' \
    --output_dir=${output_dir}  \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --optim "adamw_torch" \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_memory_MB 24000 \
    --fp16 \
    --bits 8 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. We use the [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) model for Llama.
2. There are currently some issues with evaluation, so we use `evaluation_strategy` 'no'.
3. We use the default `alpaca` template for the `prompt_template_name`. Please refer to `templates/alpaca.json` for more details.
4. For more detailed parameter information, please refer to `utils/args.py`.
5. `max_memory_MB` (default 80000) specifies the GPU memory size. You need to specify it according to your own GPU capacity.
6. We have successfully run the Llama-LoRA fine-tuning code on an `RTX3090` GPU.
7. model_name = baichuan

The corresponding script can be found at `ft_scripts/fine_baichuan.bash`.




## 5.P-Tuning Fine-tuning


### P-Tuning Fine-tuning with ChatGLM

You can use the following command to fine-tune the model using the P-Tuning method:
```bash
deepspeed --include localhost:0 src/finetuning_pt.py \
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






### Prediction of Lora
Here are some trained versions of LoRA:
* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [zhixi-13B-LoRA](https://huggingface.co/zjunlp/zhixi-13b-lora/tree/main)


Correspondence between `base_model` and `lora_weights`:
| base_model   | lora_weights   |
| ------ | ------ |
| llama-7b  | llama-7b-lora  |
| alpaca-7b | alpaca-7b-lora |
| zhixi-13b | zhixi-13b-lora |



You can use the following command to set your own parameters and execute it to make predictions using the trained LoRA model on the competition test dataset:

```bash
CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'Path or name to model' \
    --model_name 'model name' \
    --lora_weights 'Path to LoRA weights dictory' \
    --input_file 'data/valid.json' \
    --output_file 'results/results_valid.json' \
    --fp16 \
    --bits 8 
```

1. Attention!!! `--fp16` or `--bf16`, `--bits`、`--prompt_template_name` and `--model_name` must be set the same as the ones used during the [4.LoRA Fine-tuning](./README.md/#4lora-fine-tuning) process.


You can also use trained models (without Lora or with Lora merged into model parameters) to predict outputs on competition test sets:


```bash
CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'Path or name to model' \
    --model_name 'model name' \
    --input_file 'data/valid.json' \
    --output_file 'results/results_valid.json' \
    --fp16 \
    --bits 8 
```

The following models support the aforementioned approach:

[zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi) | [zjunlp/knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie)




### Prediction of P-Tuning

You can use the following command to predict the output on the competition test set using a trained P-Tuning model:
```bash
CUDA_VISIBLE_DEVICES=0 python src/inference_pt.py \
  --test_path data/valid.json \
  --device 0 \
  --ori_model_dir /model \
  --model_dir /output_dir_lora/global_step- \
  --max_len 768 \
  --max_src_len 450
```



## 6. Model Output Conversion & F1 Calculation
We provide a script, [evaluate.py](./kg2instruction/evaluate.py), to convert the model's string outputs into lists and calculate the F1 score.

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task NER \
  --language zh
```


## 7.Acknowledgment

Part of the code comes from [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)、[qlora](https://github.com/artidoro/qlora.git) many thanks.



## Citation

If you have used the code or data of this project, please refer to the following papers:
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
