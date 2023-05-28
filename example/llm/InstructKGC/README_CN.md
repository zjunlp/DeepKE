# InstructionKGC-指令驱动的自适应知识图谱构建

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README.md">English</a> | 简体中文 </b>
</p>


## 任务目标

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

## 数据

比赛数据的训练集每条数据包含如下字段：

|    字段     |                          说明                          |
| :---------: | :----------------------------------------------------: |
|     id      |                     样本唯一标识符                     |
|    input    |    模型输入文本（需要抽取其中涉及的所有关系三元组）    |
| instruction |                 模型进行抽取任务的指令                 |
| output      | 模型期望输出，以(ent1,relation,ent2)形式组成的输出文本 |
|     kg      |                  input中涉及的知识图谱                  |

在测试集中仅包含`id`、`instruction`、`input`三个字段。


## 1.准备

### 克隆代码
```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/llm/InstructKGC
mkdir result
mkdir lora
mkdir data
```

### 创建python虚拟环境, 并使用pip安装
```bash
conda create -n instructkgc python=3.9   
conda activate instructkgc
pip install -r requirements.txt
```

### 下载数据
从官网https://tianchi.aliyun.com/competition/entrance/532080/information 下载文件 `train.json` 和 `valid.json` (虽然名字是valid, 但这不是验证集, 而是比赛的测试集)并放在目录 `./data` 中.


### 模型
下面是一些模型
* [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf)
* [LLaMA-13b](https://huggingface.co/decapoda-research/llama-13b-hf)
* [Alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)
* [CaMA-13b](https://huggingface.co/zjunlp/CaMA-13B-Diff)

如果你需要使用CaMA请参考[CaMA/2.2 预训练模型权重获取与恢复](https://github.com/zjunlp/CaMA/tree/main)获得完整的CaMA模型权重。


## 2.运行


你可以通过下面的脚本使用LoRA方法来finetune模型:

注意: 由于CaMA已经在大量的信息抽取指令数据集上经过LoRA训练, 因此可以跳过这一步直接执行第3步`预测`, 你也可以选择进一步训练。

```bash
bash scripts/run_finetune.bash
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
bash scripts/run_finetune_mul.bash
```


## 3.预测

以下是训练好的一些LoRA版本:
* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [CaMA-13B-LoRA](https://huggingface.co/zjunlp/CaMA-13B-LoRA)

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


## 4.格式转换
上面的 `bash run_inference.bash` 会在 `result` 目录下输出 `output_llama_7b_e3_r8.json` 文件, 文件中不包含 'kg' 字段, 如果需要满足CCKS2023比赛的提交格式还需要从 'output' 中抽取出 'kg', 这里提供一个简单的样例 `convert.py`

```bash
python utils/convert.py \
    --pred_path "result/output_llama_7b_e3_r8.json" \
    --tgt_path "result/result_llama_7b_e3_r8.json" 
```


## 5.硬件
我们在1块 上对模型进行了finetune
注意：请确保你的设备或服务器有足够的RAM内存！！！


## 6.Acknowledgment

代码基本来自于[Alpaca-LoRA](https://github.com/tloen/alpaca-lora), 仅做了部分改动, 感谢！
