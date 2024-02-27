# InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README_CN.md">简体中文</a> </b>
</p>

- [InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction](#instructkgc-ccks2023-evaluation-of-instruction-based-knowledge-graph-construction)
  - [News](#news)
  - [🎯 1.Task Object](#-1task-object)
  - [📊 2.Data](#-2data)
    - [2.1 Existing Datasets](#21-existing-datasets)
    - [2.2Training Data Conversion](#22training-data-conversion)
    - [2.3Test Data Conversion](#23test-data-conversion)
  - [🚴 3.Preparation](#-3preparation)
    - [🛠️ 3.1Environment](#️-31environment)
    - [🐐 3.2Model](#-32model)
  - [🌰 4.LoRA Fine-tuning](#-4lora-fine-tuning)
    - [4.1 Basic Parameters](#41-basic-parameters)
    - [4.2 LoRA Fine-tuning LLaMA](#42-lora-fine-tuning-llama)
    - [4.3 LoRA Fine-tuning Alpaca](#43-lora-fine-tuning-alpaca)
    - [4.4 LoRA Fine-tuning Zhixi](#44-lora-fine-tuning-zhixi)
    - [4.5 LoRA Fine-tuning Vicuna](#45-lora-fine-tuning-vicuna)
    - [4.6 LoRA Fine-tuning ChatGLM](#46-lora-fine-tuning-chatglm)
    - [4.7 LoRA Fine-tuning Moss](#47-lora-fine-tuning-moss)
    - [4.8 LoRA Fine-tuning Baichuan](#48-lora-fine-tuning-baichuan)
    - [4.9 Continue Training with Domain-specific Data](#49-continue-training-with-domain-specific-data)
  - [🥊 5.P-Tuning Fine-tuning](#-5p-tuning-fine-tuning)
    - [5.1P-Tuning Fine-tuning with ChatGLM](#51p-tuning-fine-tuning-with-chatglm)
  - [🔴 6. Prediction](#-6-prediction)
    - [6.1 LoRA Prediction](#61-lora-prediction)
      - [6.1.1 Base Model + LoRA](#611-base-model--lora)
      - [6.1.2 IE-Specific Model](#612-ie-specific-model)
    - [6.2 P-Tuning Prediction](#62-p-tuning-prediction)
  - [🧾 7. Model Output Conversion \& F1 Calculation](#-7-model-output-conversion--f1-calculation)
  - [👋 8.Acknowledgment](#-8acknowledgment)
  - [Citation](#citation)


## News
* [2024/02] We released a large-scale (0.32B tokens) high-quality bilingual (Chinese and English) Information Extraction (IE) instruction dataset named [IEPile](https://huggingface.co/datasets/zjunlp/iepie), along with two models trained on `IEPile`, [baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) and [llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora).
* [2023/10] We released a new bilingual (Chinese and English) theme-based Information Extraction (IE) instruction dataset named [InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE) with [paper](https://arxiv.org/abs/2305.11527).
* [2023/08] We introduced a dedicated 13B model for Information Extraction (IE), named [knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie/tree/main).
* [2023/05] We initiated an instruction-based Information Extraction project.




## 🎯 1.Task Object

We define `Instruction-based KGC` as an autoregressive generation task that follows instructions. The model first needs to understand the instructions and recognize their intent. Then, based on the content of the instructions, the model extracts the corresponding triples from the input text and outputs them in the specified format. The **`instruction`** format in this paper adopts a structure similar to a JSON string, which is essentially a dictionary-type string. It consists of the following three fields:


(1) **`'instruction'`**: Task description, which outlines the task to be performed by the instruction (one of `NER`, `RE`, `EE`, `EET`, `EEA`).
(2) **`'schema'`**: A list of schemas to be extracted (`entity types`, `relation types`, `event types`).
(3) **`'input'`**: The text from which information is to be extracted.

The file [instruction.py](./ie2instruction/convert/utils/instruction.py) provides instructions for various tasks.

Below is a **data example**:

```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}
```

The data instance belongs to the `NER` task, is part of the `CoNLL2003` dataset, the schema list to be extracted includes ["`person`", "`organization`", "`else`", "`location`"], and the text to be extracted from is "*284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )*". The output is `{"person": ["Robert Allenby", "Allenby", "Miguel Angel Martin"], "organization": [], "else": [], "location": ["Australia", "Spain"]}`.

> Note that the order of schemas in the output is consistent with the order in the instruction.


<details>
  <summary><b>More Tasks Instance</b></summary>

```json
{
  "task": "EE", 
  "source": "PHEE", 
  "instruction": "{\"instruction\": \"You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.\", \"schema\": [{\"event_type\": \"potential therapeutic event\", \"trigger\": true, \"arguments\": [\"Treatment.Time_elapsed\", \"Treatment.Route\", \"Treatment.Freq\", \"Treatment\", \"Subject.Race\", \"Treatment.Disorder\", \"Effect\", \"Subject.Age\", \"Combination.Drug\", \"Treatment.Duration\", \"Subject.Population\", \"Subject.Disorder\", \"Treatment.Dosage\", \"Treatment.Drug\"]}, {\"event_type\": \"adverse event\", \"trigger\": true, \"arguments\": [\"Subject.Population\", \"Subject.Age\", \"Effect\", \"Treatment.Drug\", \"Treatment.Dosage\", \"Treatment.Freq\", \"Subject.Gender\", \"Treatment.Disorder\", \"Subject\", \"Treatment\", \"Treatment.Time_elapsed\", \"Treatment.Duration\", \"Subject.Disorder\", \"Subject.Race\", \"Combination.Drug\"]}], \"input\": \"Our findings reveal that even in patients without a history of seizures, pregabalin can cause a cortical negative myoclonus.\"}", 
  "output": "{\"potential therapeutic event\": [], \"adverse event\": [{\"trigger\": \"cause \", \"arguments\": {\"Subject.Population\": \"NAN\", \"Subject.Age\": \"NAN\", \"Effect\": \"cortical negative myoclonus\", \"Treatment.Drug\": \"pregabalin\", \"Treatment.Dosage\": \"NAN\", \"Treatment.Freq\": \"NAN\", \"Subject.Gender\": \"NAN\", \"Treatment.Disorder\": \"NAN\", \"Subject\": \"patients without a history of seizures\", \"Treatment\": \"pregabalin\", \"Treatment.Time_elapsed\": \"NAN\", \"Treatment.Duration\": \"NAN\", \"Subject.Disorder\": \"NAN\", \"Subject.Race\": \"NAN\", \"Combination.Drug\": \"NAN\"}}]}"
}

{
  "task": "RE", 
  "source": "NYT11", 
  "instruction": "{\"instruction\": \"You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.\", \"schema\": [\"neighborhood of\", \"nationality\", \"children\", \"place of death\"], \"input\": \" In the way New Jersey students know that Thomas Edison 's laboratory is in West Orange , the people of Colma know that Wyatt Earp 's ashes are buried at Hills of Eternity , a Jewish cemetery he was n't ; his wife was , and that Joe DiMaggio is at Holy Cross Cemetery , where visitors often lean bats against his gravestone . \"}", 
  "output": "{\"neighborhood of\": [], \"nationality\": [], \"children\": [], \"place of death\": [{\"subject\": \"Thomas Edison\", \"object\": \"West Orange\"}]}"
}
```

</details>

> **Note**⚠️: For the old version of the data style, please refer to [kg2instruction/README.md](./kg2instruction/README.md)


## 📊 2.Data


### 2.1 Existing Datasets

| Name | Download | Quantity | Description |
| --- | --- | --- | --- |
| InstructIE | [Google Drive](https://drive.google.com/file/d/1raf0h98x3GgIhaDyNn1dLle9_HvwD6wT/view?usp=sharing) <br/> [Hugging Face](https://huggingface.co/datasets/zjunlp/InstructIE) <br/> [ModelScope](https://modelscope.cn/datasets/ZJUNLP/InstructIE)<br/> [WiseModel](https://wisemodel.cn/datasets/zjunlp/InstructIE) | 300k+ | **Bilingual** (Chinese and English) topic-based Information Extraction (IE) instruction dataset |
| IEPile | [Google Drive](https://drive.google.com/file/d/1jPdvXOTTxlAmHkn5XkeaaCFXQkYJk5Ng/view?usp=sharing) <br/> [Hugging Face](https://huggingface.co/datasets/zjunlp/iepile) <br/> [WiseModel](https://wisemodel.cn/datasets/zjunlp/IEPile) <br/> [ModelScope](https://modelscope.cn/datasets/ZJUNLP/IEPile) | 2 million+ | Large-scale (`0.32B` tokens) high-quality **bilingual** (Chinese and English) Information Extraction (IE) instruction fine-tuning dataset |


<details>
  <summary><b>Details of InstructIE</b></summary>

**An example of a single data entry**

```json
{
  "id": "841ef2af4cfe766dd9295fb7daf321c299df0fd0cef14820dfcb421161eed4a1", 
  "text": "NGC1313 is a galaxy in the constellation of Reticulum. It was discovered by the Australian astronomer James Dunlop on September 27, 1826. It has a prominent uneven shape, and its axis does not completely revolve around its center. Near NGC1313, there is another galaxy, NGC1309.", 
  "relation": [
    {"head": "NGC1313", "head_type": "astronomical object type", "relation": "time of discovery", "tail": "September 27, 1826", "tail_type": "time"}, 
    {"head": "NGC1313", "head_type": "astronomical object type", "relation": "discoverer or inventor", "tail": "James Dunlop", "tail_type": "organization/human"}, 
    {"head": "NGC1313", "head_type": "astronomical object type", "relation": "of", "tail": "Reticulum", "tail_type": "astronomical object type"}
  ]
}
```

| Field       | Description                                                      |
| ----------- | ---------------------------------------------------------------- |
| id          | The unique identifier for each data point.                       |
| cate        | The category of the text's subject, with a total of 12 different thematic categories. |
| text       | The input text for the model, with the goal of extracting all the involved relationship triples. |
| relation    | Describes the relationship triples contained in the text, i.e., (head, head_type, relation, tail, tail_type). |


</details>


<details>
  <summary><b>Details of IEPile</b></summary>

Each instance in `IEPile` contains four fields: `task`, `source`, `instruction`, and `output`. Below are the explanations for each field:


| Field | Description |
| :---: | :---: |
| task | The task to which the instance belongs, one of the five types (`NER`, `RE`, `EE`, `EET`, `EEA`). |
| source | The dataset to which the instance belongs. |
| instruction | The instruction for inputting into the model, processed into a JSON string via json.dumps, including three fields: `"instruction"`, `"schema"`, and `"input"`. |
| output | The output in the format of a dictionary's JSON string, where the key is the schema, and the value is the extracted content. |


In `IEPile`, the **instruction** format of `IEPile` adopts a JSON-like string structure, which is essentially a dictionary-type string composed of the following three main components:
(1) **`'instruction'`**: Task description, which outlines the task to be performed by the instruction (one of `NER`, `RE`, `EE`, `EET`, `EEA`).
(2) **`'schema'`**: A list of schemas to be extracted (`entity types`, `relation types`, `event types`).
(3) **`'input'`**: The text from which information is to be extracted.

The file [instruction.py](./ie2instruction/convert/utils/instruction.py) provides instructions for various tasks.

Below is a **data example**:

```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}
```

The data instance belongs to the `NER` task, is part of the `CoNLL2003` dataset, the schema list to be extracted includes ["`person`", "`organization`", "`else`", "`location`"], and the text to be extracted from is "*284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )*". The output is `{"person": ["Robert Allenby", "Allenby", "Miguel Angel Martin"], "organization": [], "else": [], "location": ["Australia", "Spain"]}`.


</details>



### 2.2Training Data Conversion

Firstly, it's necessary to **format the data** to include `instruction` and `output` fields. For this purpose, we provide a script [convert_func.py](./ie2instruction/convert_func.py), which can batch convert data into a format that can be directly used by the model.


> Before using the [convert_func.py](./ie2instruction/convert_func.py) script, please make sure to refer to the [data](./data) directory. This directory provides detailed instructions on the data format required for each task. Refer to `sample.json` to understand the format of the data before conversion, `schema.json` to see the organization of the schema, and `train.json` to describe the data format after conversion.

> Additionally, you can directly use the bilingual (Chinese and English) information extraction dataset [zjunlp/InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE), which includes 12 themes such as characters, vehicles, works of art, natural science, man-made objects, astronomical objects, etc.


```bash
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/train.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NER \
    --split_num 6 \       
    --random_sort \
    --split train
```


* `language`: Supports two languages, `zh` (Chinese) and `en` (English), with different instruction templates used for each language.
* `task`: Currently supports five types of tasks: ['`RE`', '`NER`', '`EE`', '`EET`', '`EEA`'].
* `split_num`: Defines the maximum number of schemas that can be included in a single instruction. The default value is 4, and setting it to -1 means no splitting is done. The recommended number of task splits varies by task: **6 for NER, and 4 for RE, EE, EET, EEA**.
* `random_sort`: Whether to randomize the order of schemas in the instructions. The default is False, which means schemas are sorted alphabetically.
* `split`: Specifies the type of dataset, with options `train` or `test`.

The converted training data will contain four fields: `task`, `source`, `instruction`, `output`.


### 2.3Test Data Conversion

Before preparing the test data conversion, please visit the [data](./data) directory to understand the data structure required for each task: 1) For the input data format, see `sample.json`. 2) For the schema format, please refer to `schema.json`. 3) For the format of the transformed data, refer to `train.json`. **Unlike training data, test data input does not need to include annotation fields (`entity`, `relation`, `event`)**.


```bash
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/test.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NER \
    --split_num 6 \
    --split test
```

When setting `split` to **test**, select the appropriate number of schemas according to the task type: **6 is recommended for NER, while 4 is recommended for RE, EE, EET, EEA**. The transformed test data will contain five fields: `id`, `task`, `source`, `instruction`, `label`.

The `label` field will be used for subsequent evaluation. If the input data lacks the annotation fields (`entity`, `relation`, `event`), the transformed test data will not contain the `label` field, which is suitable for scenarios where no original annotated data is available.




## 🚴 3.Preparation


### 🛠️ 3.1Environment
Please refer to [DeepKE/example/llm/README.md](../README.md/#requirements) to create a Python virtual environment, and activate the `deepke-llm` environment:

```bash
conda activate deepke-llm
```


```bash
mkdir results
mkdir lora
mkdir data
```

Place the data in the directory `./data`


### 🐐 3.2Model 

Here are some of the models supported by the code in this repository:
[[llama](https://huggingface.co/meta-llama), [alpaca](https://github.com/tloen/alpaca-lora), [vicuna](https://huggingface.co/lmsys), [zhixi](https://github.com/zjunlp/KnowLM), [falcon](https://huggingface.co/tiiuae), [baichuan](https://huggingface.co/baichuan-inc), [chatglm](https://huggingface.co/THUDM), [qwen](https://huggingface.co/Qwen), [moss](https://huggingface.co/fnlp), [openba](https://huggingface.co/OpenBA)]




## 🌰 4.LoRA Fine-tuning

Below are some models that have been trained with ample information extraction instruction data:
* [zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) (The base model is LLaMA2-13B-Chat)
* [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) (The base model is BaiChuan2-13B-Chat)
* [zjunlp/knowlm-ie-v2](https://huggingface.co/zjunlp/knowlm-ie-v2)


### 4.1 Basic Parameters

> Important Note: All the commands below should be executed within the `IEPile` directory. For example, if you want to run the fine-tuning script, you should use the following command: `bash ft_scripts/fine_llama.bash`. Please ensure your current working directory is correct.


```bash
output_dir='lora/llama2-13b-chat-v1'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1287 src/test_finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/llama2-13b-chat' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama2' \
    --train_file 'data/train.json' \
    --valid_file 'data/dev.json' \
    --output_dir=${output_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --optim "adamw_torch" \
    --max_source_length 400 \
    --cutoff_len 700 \
    --max_target_length 300 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 
```

* `model_name`: Specifies the **name of the model architecture** you want to use (7B, 13B, Base, Chat belong to the same model architecture). Currently supported models include: ["`llama`", "`alpaca`", "`vicuna`", "`zhixi`", "`falcon`", "`baichuan`", "`chatglm`", "`qwen`", "`moss`", "`openba`"]. **Please note**, this parameter should be distinguished from `--model_name_or_path`.
* `model_name_or_path`: Model path, please download the corresponding model from [HuggingFace](https://huggingface.co/models).
* `template`: The **name of the template** used, including: `alpaca`, `baichuan`, `baichuan2`, `chatglm3`, etc. Refer to [src/datamodule/template.py](./src/datamodule/template.py) to see all supported template names. The default is the `alpaca` template. **For `Chat` versions of models, it is recommended to use the matching template, while `Base` version models can default to using `alpaca`**.
* `train_file`, `valid_file (optional)`: The **file paths** for the training set and validation set. Note: Currently, the format for files only supports **JSON format**.
* `output_dir`: The **path to save the weight parameters** after LoRA fine-tuning.
* `val_set_size`: The number of samples in the **validation set**, default is 1000.
* `per_device_train_batch_size`, `per_device_eval_batch_size`: The `batch_size` on each GPU device, adjust according to the size of the memory. For RTX3090, it is recommended to set between 2 and 4.
* `max_source_length`, `max_target_length`, `cutoff_len`: The maximum input and output lengths, and the cutoff length, which can simply be considered as the maximum input length + maximum output length. Set appropriate values according to specific needs and memory size.
* `deepspeed`: Remove if there is not enough device resources.

> Quantization can be performed by setting bits to 4; it is recommended for the RTX3090.

To learn more about parameter configuration, please refer to the [src/utils/args](./src/args). 

The specific script for fine-tuning the `LLaMA2-13B-Chat` model can be found in [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash).


### 4.2 LoRA Fine-tuning LLaMA
The specific script for fine-tuning the LLaMA model can be found in [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash).


### 4.3 LoRA Fine-tuning Alpaca
When fine-tuning the Alpaca model, you can follow the steps similar to [fine-tuning the LLaMA model](./README_CN.md/#42lora微调llama). To fine-tune, make the following **changes** to the [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash) file:

```bash
  output_dir='path to save Alpaca Lora'
  --model_name_or_path 'path or name to Alpaca' \
  --template 'alpaca' \
  --model_name 'alpaca' \
```

1. For the template, we **default to using the alpaca template**.
2. `model_name = alpaca`


### 4.4 LoRA Fine-tuning Zhixi

```bash
  output_dir='path to save Zhixi Lora'
  --model_name_or_path 'path or name to Zhixi' \
  --model_name 'zhixi' \
  --template 'alpaca' \
```

1. Since Zhixi currently only has a 13b model, it is recommended to accordingly reduce the batch size.
2. For the template, we **default to using the alpaca template**.
3. `model_name = zhixi`


### 4.5 LoRA Fine-tuning Vicuna

The corresponding script can be found in [ft_scripts/fine_vicuna.bash](./ft_scripts/fine_vicuna.bash).

1. Since the template used by Vicuna-7b-delta-v1.1 is different from the `alpaca` **template**, it is necessary to set `template vicuna`.
2. `model_name = vicuna`


### 4.6 LoRA Fine-tuning ChatGLM
The corresponding script can be found in [ft_scripts/fine_chatglm.bash](./ft_scripts/fine_chatglm.bash).

1. For the ChatGLM model, we use [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b).
2. `model_name = chatglm`
3. `template chatglm3`


### 4.7 LoRA Fine-tuning Moss

The corresponding script can be found in [ft_scripts/fine_moss.bash](./ft_scripts/fine_moss.bash).

1. For the Moss model, we use [fnlp/moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft).
2. `model_name = moss`


### 4.8 LoRA Fine-tuning Baichuan
The corresponding script can be found in [ft_scripts/fine_baichuan.bash](./ft_scripts/fine_baichuan.bash).

1. For the Baichuan model, we use [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base).
2. **Please ensure that the torch version remains at 2.0.0, otherwise issues may arise.**
3. `model_name = baichuan`
4. `template baichuan2`
5. We recommend using `--bf16`.
6. If memory overflow occurs when saving after evaluation, set `evaluation_strategy no`.


### 4.9 Continue Training with Domain-specific Data

Although the `llama2-13b-iepile-lora` and `baichuan2-13b-iepile-lora` models have undergone extensive instruction fine-tuning on multiple general datasets and thus possess a degree of **general information extraction capability**, they may still exhibit certain limitations when processing data in **specific domains** (such as `law`, `education`, `science`, `telecommunications`). To address this challenge, it is recommended to conduct **secondary training** of these models on datasets specific to these domains. This will help the models better adapt to the semantic and structural characteristics of the specific domains, enhancing their **information extraction capability within those domains**.


```bash
output_dir='lora/llama2-13b-chat-v1-continue'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1287 src/test_finetune.py \
    --do_train --do_eval \
    --overwrite_output_dir \
    --model_name_or_path 'models/llama2-13B-Chat' \
    --checkpoint_dir 'zjunlp/llama2-13b-iepile-lora' \
    --stage 'sft' \
    --model_name 'llama' \
    --template 'llama2' \
    --train_file 'data/train.json' \
    --valid_file 'data/dev.json' \
    --output_dir=${output_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --optim "adamw_torch" \
    --max_source_length 400 \
    --cutoff_len 700 \
    --max_target_length 300 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 
```

* To continue training based on the fine-tuned LoRA weights, simply point the `--checkpoint_dir` parameter to the path of the LoRA weights, for example by setting it to `'zjunlp/llama2-13b-iepile-lora'`.

> Quantization can be performed by setting bits to 4; it is recommended for the RTX3090.


> Please note that when using **`LLaMA2-IEPile`** or **`Baichuan2-IEPile`**, keep both lora_r and lora_alpha at 64. We do not provide recommended settings for these parameters.


* To continue training based on the fine-tuned model weights, just set the `--model_name_or_path` parameter to the path of the weights, such as `'zjunlp/KnowLM-IE-v2'`, without setting `--checkpoint_dir`.


The script can be found at [ft_scripts/fine_continue.bash](./ft_scripts/fine_continue.bash).



## 🥊 5.P-Tuning Fine-tuning


### 5.1P-Tuning Fine-tuning with ChatGLM

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



## 🔴 6. Prediction

### 6.1 LoRA Prediction

#### 6.1.1 Base Model + LoRA


Below are some models optimized through training with LoRA technology (**LoRA weights**):
<details>
  <summary><b>Version V1</b></summary>


* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [knowlm-13b-ie-lora](https://huggingface.co/zjunlp/knowlm-13b-ie-lora)


| checkpoint_dir | model_name_or_path | moadel_name | fp16/bf16 | template | 
| --- | --- | --- | --- | --- |
| llama-7b-lora-ie | llama-7b | llama | fp16 | alpaca |
| alpaca-7b-lora-ie | alpaca-7b | alpaca | fp16 | alpaca |
| knowlm-13b-ie-lora | zjunlp/knowlm-13b-base-v1.0 | zhixi | fp16 | alpaca |

</details>

<details>
  <summary><b>Version V1(Recommended)</b></summary>

* [zjunlp/llama2-13b-iepile-lora](https://huggingface.co/zjunlp/llama2-13b-iepile-lora/tree/main) 
* [zjunlp/baichuan2-13b-iepile-lora](https://huggingface.co/zjunlp/baichuan2-13b-iepile-lora) 
* [zjunlp/knowlm-ie-v2](https://huggingface.co/zjunlp/knowlm-ie-v2)


| checkpoint_dir | model_name_or_path | moadel_name | fp16/bf16 | template | 
| --- | --- | --- | --- | --- |
| llama2-13b-iepile-lora | LLaMA2-13B-Chat | llama | bf16 | llama2 |
| baichuan2-13b-iepile-lora | BaiChuan2-13B-Chat | baichuan | bf16 | baichuan2 |

</details>



To use these trained LoRA models for prediction, you can execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'models/llama2-13B-Chat' \
    --checkpoint_dir 'lora/llama2-13b-IEPile-lora' \
    --model_name 'llama' \
    --template 'llama2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/llama2-13b-IEPile-lora_output.json' \
    --finetuning_type lora \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --cutoff_len 512 \
    --bf16 \
    --max_new_tokens 300
```

* During inference, `model_name`, `template`, and `bf16` must be the same as the settings used during training.
* `model_name_or_path`: Specify the path to the base model being used, which must match the corresponding LoRA model.
* `checkpoint_dir`: The path to the LoRA weight files.
* `output_dir`: This parameter does not take effect during inference and any path can be specified.
* `input_file`, `output_file`: Specify the input path for the test file and the output path for the prediction results, respectively.
* `cutoff_len`, `max_new_tokens`: Set the maximum input length and the number of new tokens to be generated, adjusting according to device performance.

> Quantization can be performed by setting bits to 4; it is recommended for the RTX3090.



#### 6.1.2 IE-Specific Model
If you want to use a trained model (without LoRA or with LoRA integrated into the model parameters), you can execute the following command for prediction:

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --stage sft \
    --model_name_or_path 'models/KnowLM-IE-v2' \
    --model_name 'baichuan' \
    --template 'baichuan2' \
    --do_predict \
    --input_file 'data/input.json' \
    --output_file 'results/KnowLM-IE-v2_output.json' \
    --output_dir 'lora/test' \
    --predict_with_generate \
    --cutoff_len 512 \
    --bf16 \
    --max_new_tokens 300
```

`model_name_or_path`: The path to the weights of the model specialized for Information Extraction (IE).



### 6.2 P-Tuning Prediction

You can predict the output on the competition test set using a trained P-Tuning model with the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python src/inference_pt.py \
  --test_path data/valid.json \
  --device 0 \
  --ori_model_dir /model \
  --model_dir /output_dir_lora/global_step- \
  --max_len 768 \
  --max_src_len 450
```



## 🧾 7. Model Output Conversion & F1 Calculation

We provide scripts for evaluating the F1 scores for various tasks.

```bash
python ie2instruction/eval_func.py \
  --path1 data/NER/processed.json \
  --task NER 
```

* `task`: Currently supports five types of tasks: ['`RE`', '`NER`', '`EE`', '`EET`', '`EEA`'].
* You can set `sort_by` to `source` to calculate the F1 scores on each dataset separately.




## 👋 8.Acknowledgment

Part of the code comes from [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)、[qlora](https://github.com/artidoro/qlora.git) many thanks.



## Citation

If you have used the code or data of this project, please refer to the following papers:
```bibtex
@article{DBLP:journals/corr/abs-2305-11527,
  author       = {Honghao Gui and Shuofei Qiao and Jintian Zhang and Hongbin Ye and Mengshu Sun and Lei Liang and Huajun Chen and Ningyu Zhang},
  title        = {InstructIE: A Bilingual Instruction-based Information Extraction Dataset},
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

@article{DBLP:journals/corr/abs-2402-14710,
  author       = {Honghao Gui and
                  Hongbin Ye and
                  Lin Yuan and
                  Ningyu Zhang and
                  Mengshu Sun and
                  Lei Liang and
                  Huajun Chen},
  title        = {IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus},
  journal      = {CoRR},
  volume       = {abs/2402.14710},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2402.14710},
  doi          = {10.48550/ARXIV.2402.14710},
  eprinttype   = {arXiv},
  eprint       = {2402.14710},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2402-14710.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
