### 1、安装需要的包

注意torch和python的版本要求（torch>=1.10 ,<2.0.0,python>=3.7）。
```bash
pip install -r requirements.txt
```



### 2、数据格式转化

由于CPM-Bee的提示数据需要特定的格式，我们选择文本生成类型，将事件的trans.json转换为cpm-bee所要求的格式，并提取20%的样本作为验证集。
```
"文本生成": {"input": "今天天气很好，我和妈妈一起去公园，", "prompt": "往后写约100字", "<ans>": ""}
#数据转化前后
初始数据：{"id": 10000, "cate": "建筑", "instruction": "已知候选的关系列表：['事件', '位于', '名称由来']，请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。", "input": "浅草神社位于日本东京都台东区浅草的浅草寺本堂东侧，供奉的是土师真中知、桧田浜成、桧前武成，三位对于浅草寺创立有密切关联的人，每年5月17日都会举行三社祭。现在被指定为重要文化财产。", "output": "(浅草神社,事件,三社祭),(浅草神社,位于,浅草),(台东区,位于,东京都),(浅草寺,位于,浅草),(浅草寺,名称由来,浅草)", "kg": [["浅草神社", "事件", "三社祭"], ["浅草神社", "位于", "浅草"], ["台东区", "位于", "东京都"], ["浅草寺", "位于", "浅草"], ["浅草寺", "名称由来", "浅草"]]}
转化后：{ "input": "浅草神社位于日本东京都台东区浅草的浅草寺本堂东侧，供奉的是土师真中知、桧田浜成、桧前武成，三位对于浅草寺创立有密切关联的人，每年5月17日都会举行三社祭。现在被指定为重要文化财产。", "prompt": "已知候选的关系列表：['事件', '位于', '名称由来']，请你根据关系列表，从以下输入中抽取出可能存在的头实体(Subject)与尾实体(Object)，并给出对应的关系三元组。请按照 (Subject,Relation,Object) 的格式回答。", "<ans>": "(浅草神社,事件,三社祭),(浅草神社,位于,浅草),(台东区,位于,东京都),(浅草寺,位于,浅草),(浅草寺,名称由来,浅草)"}
```
```bash
python data_reformate.py
```

将处理好的数据放入bee_data文件夹，并用preprocess_dataset.py提供的数据处理方法将其转为二进制文件

```bash
python preprocess_dataset.py --input bee_data --output_path bin_data --output_name ccpm_data
```


### 3、模型微调
从[CPM-Bee](https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune)下载CPM-Bee模型，你可以选择1B,2B,5B,10B版本，并根据模型文件的路径地址更新finetune_cpm_bee.sh。

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
```



运行脚本

```
bash scripts/finetune_cpm_bee.sh
```
如果运行没有问题，你可以在上面的bash中设置的路径中获得微调后的模型。
### 4.文本推理
使用提供的文本生成工具来生成事件文本，你可以得到cpm_bee_TG.json
```bash
python text_generation.py
```
### 5.比赛数据格式转换
上面的bash run_inference.bash会在结果目录中输出没有kg'字段的cpm_bee_TG.json，如果你需要满足CCKS2023的提交格式，你需要从output'中提取kg，下面是一个简单的例子convert.py
```bash
python ../InstructKGC/utils/convert.py
-pred_path "cpm_bee_TG.json"
-tgt_path "cpm_bee_TG_kg.json'
```