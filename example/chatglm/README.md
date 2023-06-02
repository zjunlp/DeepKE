# InstructionKGC-指令驱动的自适应知识图谱构建

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/master/chatglm/README_EN.md">English</a> | 简体中文 </b>
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


### 创建python虚拟环境, 并使用pip安装
```bash
conda create -n chatglm python=3.9   
conda activate chatglm
pip install -r requirements.txt
```

### 下载数据
从官网https://tianchi.aliyun.com/competition/entrance/532080/information 下载文件 `train.json` 和 `valid.json` (虽然名字是valid, 但这不是验证集, 而是比赛的测试集)并放在目录 `./data` 中.


## 2.运行


你可以通过下面的命令使用LoRA方法来finetune模型:


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed finetuning_lora.py --num_train_epochs 5 --train_batch_size 2 --lora_r 8
```

或者是在finetuning_lora函数中设置自己的参数执行:

```bash
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/train.json', type=str, help='')
    parser.add_argument('--model_dir', default="./model",  type=str, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--train_batch_size', default=1, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_lora/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=400, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument('--prompt_text', type=str,default="",help='')
    return parser.parse_args()

```




## 3.预测

你可以通过下面的命令使用训练好的LoRA模型在比赛测试集上预测输出:

```bash
CUDA_VISIBLE_DEVICES=0 python predict_lora.py 
```

或者是在predict_lora函数中设置自己的参数执行::

```bash
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/test.json', type=str, help='')
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--model_dir',default="./model", type=str,help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--prompt_text', type=str,default="",help='')
```



## 4.硬件
我们在4块3090上对模型进行了finetune,在一块3090上进行了预测
注意：请确保你的设备或服务器有足够的RAM内存！！！


## 5.Acknowledgment

代码基本来自于[ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning), 仅做了部分改动, 感谢！
