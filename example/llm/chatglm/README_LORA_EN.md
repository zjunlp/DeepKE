# InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README_CN.md">简体中文</a> </b>
</p>


## Task Object

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

## Data

The training dataset for the competition contains the following fields for each data entry:

|    Field    |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|     id      |                   Sample unique identifier                   |
|    input    | Model input text (need to extract all triples involved within) |
| instruction |   Instruction for the model to perform the extraction task   |
|    output   | Expected model output, in the form of output text composed of (ent1, relation, ent2) |
|     kg      |             Knowledge graph involved in the input             |

In the test set, only the three fields `id`, `instruction`, and `input` are included.


## 1. Preparation

### Create a Python virtual environment and install using pip

```bash
conda create -n chatglm python=3.9   
conda activate chatglm
pip install -r requirements.txt
```


### Download data
Download  `train.json` and `valid.json`  (although the name is valid, this is not a validation set, but a test set for the competition) from official website https://tianchi.aliyun.com/competition/entrance/, and place them in the directory `./data`

## 2. Run

You can use the LoRA method to finetune the model using the following script:



```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed finetuning_lora.py --num_train_epochs 5 --train_batch_size 2 --lora_r 8
```

Alternatively, set your own parameters to execute using the following command:

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



## 3. Predict or Inference

You can use the trained LoRA model to predict the output on the competition test set using the following script:

```bash
CUDA_VISIBLE_DEVICES=0 python predict_lora.py 
```

Alternatively, set your own parameters to execute using the following command:

```bash
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/test.json', type=str, help='')
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--model_dir',default="./model", type=str,help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--prompt_text', type=str,default="",help='')
```

## 4. Hardware
We performed finetune on the model on 4 `RTX3090 24GB`,We predict finetune on the model on 1 `RTX3090 24GB`
Attention: Please ensure that your device or server has sufficient RAM memory!!!


## 6.Acknowledgment
The code basically comes from [ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning). Only some changes have been made, many thanks.
