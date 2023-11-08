# InstructionKGC-指令驱动的自适应知识图谱构建

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README.md">English</a> | 简体中文 </b>
</p>


- [InstructionKGC-指令驱动的自适应知识图谱构建](#instructionkgc-指令驱动的自适应知识图谱构建)
  - [1.任务目标](#1任务目标)
  - [2.数据](#2数据)
    - [2.1数据预处理](#21数据预处理)
    - [2.2现有数据集](#22现有数据集)
  - [3.准备](#3准备)
    - [3.1环境](#31环境)
    - [3.2下载数据](#32下载数据)
    - [3.3模型](#33模型)
  - [4.LoRA微调](#4lora微调)
    - [4.1基础参数](#41基础参数)
    - [4.2LoRA微调LLaMA](#42lora微调llama)
    - [4.3LoRA微调Alpaca](#43lora微调alpaca)
    - [4.4LoRA微调智析](#44lora微调智析)
    - [4.5LoRA微调Vicuna](#45lora微调vicuna)
    - [4.6LoRA微调ChatGLM](#46lora微调chatglm)
    - [4.7LoRA微调Moss](#47lora微调moss)
    - [4.8LoRA微调Baichuan](#48lora微调baichuan)
  - [5.P-Tuning微调](#5p-tuning微调)
    - [5.1P-Tuning微调ChatGLM](#51p-tuning微调chatglm)
  - [6.预测](#6预测)
    - [6.1LoRA预测](#61lora预测)
      - [6.1.1基础模型+Lora](#611基础模型lora)
      - [6.1.2IE专用模型](#612ie专用模型)
    - [6.2P-Tuning预测](#62p-tuning预测)
  - [7.模型输出转换\&计算F1](#7模型输出转换计算f1)
  - [8.Acknowledgment](#8acknowledgment)
  - [Citation](#citation)



## 1.任务目标

根据用户输入的指令抽取相应类型的实体和关系，构建知识图谱。其中可能包含知识图谱补全任务，即任务需要模型在抽取实体关系三元组的同时对缺失三元组进行补全。

以下是一个**知识图谱构建任务**例子，输入一段文本`input`和`instruction`（包括想要抽取的实体类型和关系类型），以`(ent1,rel,ent2)`的形式输出`input`中包含的所有关系三元组`output`：

```python
instruction="使用自然语言抽取三元组,已知下列句子,请从句子中抽取出可能的实体、关系,抽取实体类型为{'专业','时间','人类','组织','地理地区','事件'},关系类型为{'体育运动','包含行政领土','参加','国家','邦交国','夺得','举办地点','属于','获奖'},你可以先识别出实体再判断实体之间的关系,以(头实体,关系,尾实体)的形式回答"
input="2006年，弗雷泽出战中国天津举行的女子水球世界杯，协助国家队夺得冠军。2008年，弗雷泽代表澳大利亚参加北京奥运会女子水球比赛，赢得铜牌。"
output="(弗雷泽,获奖,铜牌)(女子水球世界杯,举办地点,天津)(弗雷泽,属于,国家队)(弗雷泽,国家,澳大利亚)(弗雷泽,参加,北京奥运会女子水球比赛)(中国,包含行政领土,天津)(中国,邦交国,澳大利亚)(北京奥运会女子水球比赛,举办地点,北京)(女子水球世界杯,体育运动,水球)(国家队,夺得,冠军)"
```



## 2.数据


### 2.1数据预处理
在对模型进行数据输入之前，需要将数据格式化以包含`instruction`和`input`字段。为此，我们提供了一个脚本 [kg2instruction/convert.py](./kg2instruction/convert.py)，它可以将数据批量转换成模型可以直接使用的格式。

> 在使用 [kg2instruction/convert.py](./kg2instruction/convert.py) 脚本之前，请确保参考了 [data](./data) 目录。该目录中详细列出了每种任务所需的数据格式要求。


```bash              
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \      # 指定转换脚本和模板使用的语言, ['zh', 'en']
  --task NER \         # 指定任务类型：['RE', 'NER', 'EE', 'EET', 'EEA'] 中的一种
  --sample -1 \        # 如果为-1，则随机采样20种指令和4种输出格式中的一种；如果为指定数值，则使用对应的指令格式，取值范围为 -1<=sample<20
  --neg_ratio 1 \      # 设置所有样本的负采样比例
  --neg_schema 1 \     # 设置从schema中负采样的比例
  --random_sort        # 是否对指令中的schema列表进行随机排序
```


对于测试数据，可以使用 [kg2instruction/convert_test.py](./kg2instruction/convert_test.py) 脚本，它不要求数据包含标签（`entity`、`relation`、`event`）字段，只需提供input字段和相应的schema_path。

```bash
python kg2instruction/convert_test.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \
  --task NER \
  --sample 0
```


* 更详细的关于IE模版、数据格式转换、数据提取请参考[kg2instruction/README_CN.md](./kg2instruction/README_CN.md)
* 您也可以自行构建包含`instruction`和`input`字段的数据



### 2.2现有数据集
下面是一些现成的处理后的数据：

| 名称                  | 下载                                                                                                                     | 数量     | 描述                                                                                                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| InstructIE-train       | [Google drive](https://drive.google.com/file/d/1VX5buWC9qVeVuudh_mhc_nC7IPPpGchQ/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [百度云盘](https://pan.baidu.com/s/1xXVrjkinw4cyKKFBR8BwQw?pwd=x4s7)    | 30w+ | InstructIE训练集                                                                                     |
| InstructIE-valid       | [Google drive](https://drive.google.com/file/d/1EMvqYnnniKCGEYMLoENE1VD6DrcQ1Hhj/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [百度云盘](https://pan.baidu.com/s/11u_f_JT30W6B5xmUPC3enw?pwd=71ie)    | 2000+ | InstructIE验证集                                                                                     |
| InstructIE-test       | [Google drive](https://drive.google.com/file/d/1WdG6_ouS-dBjWUXLuROx03hP-1_QY5n4/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [百度云盘](https://pan.baidu.com/s/1JiRiOoyBVOold58zY482TA?pwd=cyr9)     | 2000+ | InstructIE测试集                                                                                     |
| train.json, valid.json          | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing)                     | 5000   | [CCKS2023 开放环境下的知识图谱构建与补全评测任务一：指令驱动的自适应知识图谱构建](https://tianchi.aliyun.com/competition/entrance/532080/introduction) 中的初赛训练集及测试集 |



`InstructIE-train` 数据集包含两个核心文件：`InstructIE-zh.json` 和 `InstructIE-en.json`。这两个文件都涵盖了丰富的字段，用于详细描述数据集的不同方面：

- `'id'`：每条数据的唯一标识符，确保数据项的独立性和可追踪性。
- `'cate'`：文本主题分类，为文本内容提供了一个高级的分类标签（共有12种主题）。
- `'entity'`和`'relation'`：分别代表实体和关系三元组，这些字段允许用户自由构建信息抽取的指令和预期输出结果。

对于验证集`InstructIE-valid`和测试集`InstructIE-test`，它们包含中英双语版本，保证了数据集在不同语言环境下的适用性。

- `train.json`：这个文件中的字段定义与`InstructIE-train`一致，但`'instruction'`和`'output'`字段展示了一种格式。尽管如此，用户仍可以依据`'relation'`字段自由构建信息抽取的指令和输出。
- `valid.json`：其字段意义与`train.json`保持一致，但此数据集经过众包标注处理，提供了更高的准确性和可靠性。

各字段详细说明如下：

|    字段     |                             说明                             |
| :---------: | :----------------------------------------------------------: |
|     id      |                       每个数据点的唯一标识符。                       |
|    cate     |           文本的主题类别，总计12种不同的主题分类。               |
|    input    | 模型的输入文本，目标是从中抽取涉及的所有关系三元组。                  |
| instruction |                 指导模型执行信息抽取任务的指示。                    |
|    output   |                      模型的预期输出结果。                        |
|   entity    |            描述实体以及其对应类型的详细信息(entity, entity_type)。    |
|  relation   |   描述文本中包含的关系三元组，即实体间的联系(head, relation, tail)。   |

利用上述字段，用户可以灵活地设计和实施针对不同信息抽取需求的指令和输出格式。

此处 [schema](./kg2instruction/convert/utils.py) 提供了12种文本主题, 以及该主题下常见的关系类型。





## 3.准备



### 3.1环境
在开始之前，请确保根据[DeepKE/example/llm/README_CN.md](../README_CN.md/#环境依赖)中的指导创建了适当的Python虚拟环境。创建并配置好虚拟环境后，请通过以下命令激活名为deepke-llm的环境：

```bash
conda activate deepke-llm
```

> 为了确保与qlora技术的兼容性，我们对deepke-llm环境中的几个关键库进行了版本更新

1. transformers 0.17.1 -> 4.30.2
2. accelerate 4.28.1 -> 0.20.3
3. bitsandbytes 0.37.2 -> 0.39.1
4. peft 0.2.0 -> 0.4.0

请确保您的环境中这些库的版本与上述要求相匹配，以便顺利运行接下来的任务。



### 3.2下载数据
```bash
mkdir results
mkdir lora
mkdir data
```

数据放在目录 `./data` 中。


### 3.3模型
下面是一些模型
* [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | [LLaMA-13b](https://huggingface.co/decapoda-research/llama-13b-hf)
* [Alpaca-7b](https://huggingface.co/circulus/alpaca-7b) | [Alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)
* [Vicuna-7b-delta-v1.1](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) | [Vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) | 
* [zjunlp/knowlm-13b-base-v1.0](https://huggingface.co/zjunlp/knowlm-13b-base-v1.0)(需搭配相应的IE Lora) | [zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi)(无需Lora即可直接预测) | [zjunlp/knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie)(无需Lora, IE能力更强, 但通用性有所削弱)
* [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
* [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
* [Chinese-LLaMA-7B](https://huggingface.co/Linly-AI/Chinese-LLaMA-7B)
* [baichuan-inc/Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B) | [baichuan-inc/Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) | [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) | [baichuan-inc/Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)



## 4.LoRA微调

### 4.1基础参数
进行LoRA微调时，您需要配置一些基础参数来指定模型类型、数据集路径以及输出设置等。以下是可用的基础参数及其说明：

* `--model_name`: 指定您想要使用的模型名称。当前支持的模型列表包括：["llama", "falcon", "baichuan", "chatglm", "moss", "alpaca", "vicuna", "zhixi"]。请注意，此参数应与model_name_or_path保持区分。
* `--train_file` 和 `--valid_file`（可选）: 分别指向您的训练集和验证集的json格式文件路径。如果未提供 valid_file，系统将默认从 train_file 指定的文件中划分出 val_set_size 指定数量的样本作为验证集。您也可以通过调整 val_set_size 参数来改变验证集的样本数量。
* `--output_dir``: 设置LoRA微调后的权重参数保存路径。
* `--val_set_size`: 定义验证集的样本数量，默认为1000。
* `--prompt_template_name`: 选择使用的模板名称。目前支持三种模板类型：[alpaca, vicuna, moss]，默认使用的是alpaca模板。
* `--max_memory_MB`（默认设置为80000）用以指定GPU显存的大小。请根据您的GPU性能来进行相应调整。
* 要了解更多关于参数配置的信息，请参考 [src/utils/args.py](./src/utils/args.py) 文件。

> 重要提示：以下的所有命令均应在InstrctKGC目录下执行。例如，如果您想运行微调脚本，您应该使用如下命令：bash scripts/fine_llama.bash。请确保您的当前工作目录正确。


### 4.2LoRA微调LLaMA

要使用LoRA技术微调LLaMA模型，您可以设置自定义参数并运行以下命令：

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
    --bits 4 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. Llama模型我们采用[LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf)
2. 对于prompt_template_name，我们默认使用alpaca模板。模板的详细内容可以在 [templates/alpaca.json](./templates/alpaca.json) 文件中找到。
3. 我们已经在RTX3090 GPU上成功运行了LLaMA模型使用LoRA技术的微调代码。
4. model_name = llama（llama2也是llama）

微调LLaMA模型的具体脚本可以在 [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash) 中找到。



### 4.3LoRA微调Alpaca

微调Alpaca模型时，您可遵循与[微调LLaMA模型](./README_CN.md/#42lora微调llama)类似的步骤。要进行微调，请对[ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash)文件做出以下修改：


```bash
output_dir='path to save Alpaca Lora'
--model_name_or_path 'path or name to Alpaca' \
--model_name 'alpaca' \
```

1. Alpaca模型我们采用[Alpaca-7b](https://huggingface.co/circulus/alpaca-7b)
2. 对于prompt_template_name，我们默认使用alpaca模板。模板的详细内容可以在 [templates/alpaca.json](./templates/alpaca.json) 文件中找到。
3. 我们已经在RTX3090 GPU上成功运行了Alpaca模型使用LoRA技术的微调代码。
4. model_name = alpaca



### 4.4LoRA微调智析

在开始微调智析模型之前，请确保遵循[KnowLM2.2预训练模型权重获取与恢复](https://github.com/zjunlp/KnowLM#2-2)的指南获取完整的智析模型权重。

**重要提示**：由于智析模型已经在丰富的信息抽取任务数据集上进行了LoRA训练，你可能不需要再次微调，可以直接进行预测任务。如果选择进行进一步训练，可以按照以下步骤操作。

微调智析模型的指令与[微调LLaMA模型](./README_CN.md/#42lora微调llama)类似，只需在[ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash)中作如下调整：


```bash
output_dir='path to save Zhixi Lora'
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--model_name_or_path 'path or name to Zhixi' \
--model_name 'zhixi' \
```

1. 由于Zhixi目前只有13b的模型, 建议相应地减小批处理大小batch size
2. 对于prompt_template_name，我们默认使用alpaca模板。模板的详细内容可以在 [templates/alpaca.json](./templates/alpaca.json) 文件中找到。
3. 我们已经在RTX3090 GPU上成功运行了Zhixi模型使用LoRA技术的微调代码。
4. model_name = zhixi



### 4.5LoRA微调Vicuna

你可以通过下面的命令设置自己的参数使用LoRA方法来微调Vicuna模型:

```bash
output_dir='path to save Vicuna Lora'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --master_port=1331 src/finetune.py \
    --do_train --do_eval \
    --model_name_or_path 'path or name to Vicuna' \
    --model_name 'vicuna' \
    --prompt_template_name 'vicuna' \
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
    --bits 4 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. Vicuna模型我们采用[Vicuna-7b-delta-v1.1](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1)
2. 由于Vicuna-7b-delta-v1.1所使用的prompt_template_name与`alpaca`模版不同, 因此需要设置 `--prompt_template_name 'vicuna'`, 详见 [templates/vicuna.json](./templates//vicuna.json)
3. 我们在 `RTX3090` 上跑通了vicuna-lora微调代码
4. model_name = vicuna

相应的脚本在 [ft_scripts/fine_vicuna.bash](./ft_scripts//fine_vicuna.bash)



### 4.6LoRA微调ChatGLM

你可以通过下面的命令设置自己的参数使用LoRA方法来微调ChatGLM模型，注意⚠️目前chatglm更新速度较快，请您确保您的模型与chatglm最新模型保持一致:
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

1. ChatGLM模型我们采用[THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
2. `prompt_template_name`我们采用默认的`alpaca`模版, 详见 [templates/alpaca.json](./templates/alpaca.json)
3. 由于使用8bits量化后训练得到的模型效果不佳, 因此对于ChatGLM我们没有采用量化策略
4. model_name = chatglm

相应的脚本在 [ft_scripts/fine_chatglm.bash](./ft_scripts//fine_chatglm.bash)




### 4.7LoRA微调Moss

你可以通过下面的命令设置自己的参数使用LoRA方法来微调Moss模型:

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

1. Moss模型我们采用[moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
2. prompt_template_name在alpaca模版的基础上做了一些修改, 详见 [templates/moss.json](./templates/moss.json), 因此需要设置 `--prompt_template_name 'moss'`
3. 由于 `RTX3090` 显存限制, 我们采用`qlora`技术进行4bits量化, 你也可以在`V100`、`A100`上尝试8bits量化和不量化策略
4. 我们在 `RTX3090` 上跑通了moss-lora微调代码
5. model_name = moss

相应的脚本在 [ft_scripts/fine_moss.bash](./ft_scripts/fine_moss.bash)




### 4.8LoRA微调Baichuan

你可以通过下面的命令设置自己的参数使用LoRA方法来微调Baichuan模型:

```bash
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --master_port=1331 src/finetune.py \
    --do_train --do_eval \
    --model_name_or_path 'path or name to Baichuan' \
    --model_name 'baichuan' \
    --train_file 'data/train.json' \
    --output_dir=${output_dir}  \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --optim "adamw_torch" \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_memory_MB 24000 \
    --fp16 \
    --bits 4 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. Baichuan模型我们采用[baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
2. 目前在evaluation方面存在一些问题, 因此我们使用`evaluation_strategy` "no"
3. `prompt_template_name`我们采用默认的`alpaca`模版, 详见 [templates/alpaca.json](./templates/alpaca.json)
4. 我们在 `RTX3090` 上跑通了baichuan-lora微调代码
5. model_name = baichuan


相应的脚本在 [ft_scripts/fine_baichuan.bash](./ft_scripts/fine_baichuan.bash)



## 5.P-Tuning微调


### 5.1P-Tuning微调ChatGLM
你可以通过下面的命令使用P-Tuning方法来finetune模型:
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




## 6.预测


### 6.1LoRA预测

#### 6.1.1基础模型+Lora
以下是一些经过LoRA技术训练优化的模型(Lora权重)：
* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [zhixi-13b-lora](https://huggingface.co/zjunlp/zhixi-13b-lora/tree/main)

以下表格显示了基础模型和其对应的LoRA权重之间的关系：

| 基础模型                 | LoRA权重            |
| ----------------------- | ------------------- |
| llama-7b                | llama-7b-lora-ie       |
| alpaca-7b               | alpaca-7b-lora-ie      |
| zjunlp/knowlm-13b-base-v1.0 | zhixi-13b-lora      |


要使用这些训练好的LoRA模型进行预测，可以执行以下命令：

```bash
CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path '模型路径或名称' \
    --model_name '模型名称' \
    --lora_weights 'LoRA权重的路径' \
    --input_file 'data/valid.json' \
    --output_file 'results/results_valid.json' \
    --fp16 \
    --bits 4 
```


**注意**：请确保`--fp16` 或 `--bf16`、`--bits`、`--prompt_template_name`、`--model_name`的设置与[4.LoRA微调](./README_CN.md/#4lora微调)时保持一致。


#### 6.1.2IE专用模型
若要使用已训练的模型（无LoRA或LoRA已集成到模型参数中），可以执行以下命令进行预测：

```bash
CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path '模型路径或名称' \
    --model_name '模型名称' \
    --input_file 'data/valid.json' \
    --output_file 'results/results_valid.json' \
    --fp16 \
    --bits 4 
```

以下模型适用上述预测方法：
[zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi) | [zjunlp/knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie)




### 6.2P-Tuning预测

你可以通过下面的命令使用训练好的P-Tuning模型在比赛测试集上预测输出:
```bash
CUDA_VISIBLE_DEVICES=0 python src/inference_pt.py \
  --test_path data/valid.json \
  --device 0 \
  --ori_model_dir /model \
  --model_dir /output_dir_lora/global_step- \
  --max_len 768 \
  --max_src_len 450
```





## 7.模型输出转换&计算F1
我们提供 [evaluate.py](./kg2instruction/evaluate.py) 的脚本，用于将模型的字符串输出转换为列表并计算 F1 分数。

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task NER \
  --language zh
```


## 8.Acknowledgment

部分代码来自于 [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)、[qlora](https://github.com/artidoro/qlora.git), 感谢！


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
