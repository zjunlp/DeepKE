### 1.Install all libs needed 

Notice   torch and python version requirements(torch>=1.10 ,<2.0.0,python>=3.7)
```bash
pip install -r requirements.txt
```

### 2.Data reformat

Since the CPM-Bee prompt data needs to be in a specific format,we choose  the case Text  generation and convert the event's trans.json into the format required by cpm-bee and extract 20% of the sample to be used as validation set.
CPM-Bee Text generation format
```
"文本生成": {"input": "今天天气很好，我和妈妈一起去公园，", "prompt": "往后写约100字", "<ans>": ""}
#数据转化前后
初始数据：{"id": 10000, "cate": "建筑", "instruction": "已知候选的关系列表：['事件', '位于', '名称由来']，请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。", "input": "浅草神社位于日本东京都台东区浅草的浅草寺本堂东侧，供奉的是土师真中知、桧田浜成、桧前武成，三位对于浅草寺创立有密切关联的人，每年5月17日都会举行三社祭。现在被指定为重要文化财产。", "output": "(浅草神社,事件,三社祭),(浅草神社,位于,浅草),(台东区,位于,东京都),(浅草寺,位于,浅草),(浅草寺,名称由来,浅草)", "kg": [["浅草神社", "事件", "三社祭"], ["浅草神社", "位于", "浅草"], ["台东区", "位于", "东京都"], ["浅草寺", "位于", "浅草"], ["浅草寺", "名称由来", "浅草"]]}
转化后：{ "input": "浅草神社位于日本东京都台东区浅草的浅草寺本堂东侧，供奉的是土师真中知、桧田浜成、桧前武成，三位对于浅草寺创立有密切关联的人，每年5月17日都会举行三社祭。现在被指定为重要文化财产。", "prompt": "已知候选的关系列表：['事件', '位于', '名称由来']，请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。", "<ans>": "(浅草神社,事件,三社祭),(浅草神社,位于,浅草),(台东区,位于,东京都),(浅草寺,位于,浅草),(浅草寺,名称由来,浅草)"}
```
```bash
python data_reformate.py
```
put train.jsonl ,eval.jsonl in bee_data, and use the data process tool to conversion to binary format
```bash
python preprocess_dataset.py --input bee_data --output_path bin_data --output_name ccpm_data
```
### 3.Delta-tuning
download the CPM-Bee model from [CPM-Bee](https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune) ,you can choose 1B,2B,5B,10B verson as you like and update finetune_cpm_bee.sh  according to the path address of the model file
```bash
#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12345

OPTS=""
OPTS+=" --use-delta"
OPTS+=" --model-config config/cpm-bee-10b.json"
OPTS+=" --dataset path/to/dataset"
OPTS+=" --eval_dataset path/to/eval/dataset"
OPTS+=" --epoch 100"
OPTS+=" --batch-size 5"
OPTS+=" --train-iters 100"
OPTS+=" --save-name cpm_bee_finetune"
OPTS+=" --max-length 2048"
OPTS+=" --save results/"
OPTS+=" --lr 0.0001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 1"
OPTS+=" --eval-interval 1000"
OPTS+=" --early-stop-patience 5"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 32768"
OPTS+=" --start-step 0"
OPTS+=" --load path/to/your/model.pt"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} finetune_cpm_bee.py ${OPTS}"

echo ${CMD}
$CMD


```
run the bash
```bash
bash scripts/finetune_cpm_bee.sh
```

if it runs without faults,you can get the checkpoint of model in the path you set in the bash above
### 4.Text-Generation

Use the text generation tool provided to generate the event text and you can get cpm_bee_TG.json
```bash
python text_generation.py
```
### 5.Competition format conversion

The above bash run_inference.bash will output cpm_bee_TG.json in the result directory without the kg' field, if you need to meet the CCKS2023 submission format you will need to extract the kg from output', here is a simple example convert.py
```bash
python ../InstructKGC/convert.py
-pred_path "cpm_bee_TG.json"
-tgt_path "cpm_bee_TG_kg.json'
```