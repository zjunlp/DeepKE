# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/triple/mt5/README_ZH.md">简体中文</a> </b>
</p>


## 1. Preparation

### Clone Code

```bash
git clone  https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/triple/mt5
```


### Create a Python virtual environment and install it using pip

```bash
conda create -n ccks-mt5 python=3.9   
conda activate ccks-mt5
pip install -r requirements.txt
```


### Download data
Download from official website https://tianchi.aliyun.com/competition/entrance/532080/information Download the files' train. json 'and' valid. json 'and place them in the directory/ In data


## 2. Run

You can start the finetune model using the following script:

```bash
bash run_ finetene_ ds.bash
```

Alternatively, set your own parameters to execute using the following command:

```bash
deepspeed  --include localhost:0,1 run_ finetune.py \
--do_ train --do_ eval --do_ predict \
--num_ train_ epochs 10 \
--per_ device_ train_ batch_ size 16 \
--per_ device_ eval_ batch_ size 48 \
--gradient_ accumulation_ steps 2 \
--predict_ with_ generate \
--from_ checkpoint=True \
--overwrite_ output_ dir=False \
--model_ name_ or_ path google/mt5-base   \
--output_ dir output/ccks_ mt5-base_ f1_ 1e-4  \
--logging_ dir output/ccks_ mt5-base_ f1_ 1e-4_ log \
--train_ file data/train.json \
--test_ file data/valid.json \
--save_ total_ limit 1 \
--load_ best_ model_ at_ end \
--save_ strategy "epoch" \
--evaluation_ strategy "epoch" \
--metric_ for_ best_ model "overall-score" \
--learning_ rate 1e-4 \
--use_ fast_ tokenizer=True \
--preprocessing_ num_ workers 4 \
--generation_ max_ length 256 \
--generation_ num_ beams 1 \
--gradient_ checkpointing=True \
--deepspeed "configs/ds_mt5_z3_config_bf16.json" \
--seed 42 \
--bf16=True \
```

1.In DeepSpeed, `-- include localhost: 0,1` needs to be used instead of `'CUDA_ VISIBLE_ DEFICES="0,1" `, the effect is the same
2. For different runs (different settings), you need to modify the parameter `output'_ dir`、`logging_ Dir`, otherwise the new run will overwrite the old run 
3. For more information on parameters, please refer to `./arguments. py` and [Transformers:TrainingArgument](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/trainer#transformers.TrainingArguments). Please review the configuration of DeepSpeed https://www.deepspeed.ai/docs/config-json/


## 3. Hardware
We conducted finetune on the model on two pieces of 'RTX3090 24GB' If your device configuration is higher or lower, please choose a different scale model or adjust parameters ` batch_ size`, `gradient_ accumulation_ steps`.



## 4.Result
After training 10 epochs, the final results obtained are as follows:
| model    | Score |
| -------- | ----- |
| mt5-base | 54.32 |



## 5.Acknowledgment
Part of the code comes from [deepseed plan t5 summarization. ipynb](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/deepseed-flan-t5-summarization.ipynb) many thanks.

