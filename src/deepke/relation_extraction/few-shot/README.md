# KnowPrompt


Code and datasets for our paper "KnowPrompt: Knowledge-aware Prompt-tuning  with  Synergistic Optimization for Relation Extraction"

Requirements
==========
To install requirements:

```
pip install -r requirements.txt
```

Datasets
==========

We provide all the datasets and prompts used in our experiments.

+ [[SEMEVAL]](dataset/semeval)

+ [[DialogRE]](../datasets/dialogue)

+ [[TACRED-Revisit]](../datasets/tacrev)

+ [[Re-TACRED]](../datasets/retacred)

+ [[Wiki80]](../datasets/wiki80)

The expected structure of files is:


```
knowprompt
 |-- dataset
 |    |-- semeval
 |    |    |-- train.txt       
 |    |    |-- dev.txt
 |    |    |-- test.txt
 |    |    |-- temp.txt
 |    |    |-- rel2id.json
 |    |-- dialogue
 |    |    |-- train.json       
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- rel2id.json
 |    |-- wiki80
 |    |    |-- train.txt       
 |    |    |-- dev.txt
 |    |    |-- test.txt
 |    |    |-- temp.txt
 |    |    |-- rel2id.json
 |    |-- tacrev
 |    |    |-- train.txt       
 |    |    |-- dev.txt
 |    |    |-- test.txt
 |    |    |-- temp.txt
 |    |    |-- rel2id.json
 |    |-- retacred
 |    |    |-- train.txt       
 |    |    |-- dev.txt
 |    |    |-- test.txt
 |    |    |-- temp.txt
 |    |    |-- rel2id.json
 |-- scripts
 |    |-- semeval.sh
 |    |-- dialogue.sh
 |    |-- ...
 
```


Run the experiments
==========

## Initialize the answer words

Use the comand below to get the answer words to use in the training.

```shell
python get_label_word.py --model_name_or_path bert-large-uncased --dataset_name semeval
```

The `{answer_words}.pt`will be saved in the dataset, you need to assign the `model_name_or_path` and `dataset_name` in the `get_label_word.py`.

## Split dataset

Download the data first, and put it to `dataset` folder. Run the comand below, and get the few shot dataset.

```shell
python generate_k_shot.py --data_dir ./dataset --k 8 --dataset semeval
cd dataset/semeval
cp rel2id.json val.txt test.txt ./k-shot/8-1
```

You need to modify the `k` and `dataset` to assign k-shot and dataset. Here we default seed as 1,2,3,4,5 to split each k-shot, you can revise it in the `generate_k_shot.py`

## Let's run

Our script code can automatically run the experiments in 8-shot, 16-shot, 32-shot and 
standard supervised settings with both the procedures of train, eval and test. We just choose the random seed to be 1 as an example in our code. Actually you can perform multiple experments with different seeds.

#### Example for SEMEVAL
Train the KonwPrompt model on SEMEVAL with the following command:

```bash
>> bash scripts/semeval.sh  # for bert-large-uncased
```
As the scripts  for `TACRED-Revist`, `Re-TACRED`, `Wiki80` included in our paper are also provided, you just need to run it like above example.

#### Example for DialogRE
As the data format of DialogRE is very different from other dataset, Class of processor is also different. 
Train the KonwPrompt model on DialogRE with the following command:

```bash
>> bash scripts/dialogue.sh  # for bert-base-uncased
```