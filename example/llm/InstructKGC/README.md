# InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC/README_CN.md">简体中文</a> </b>
</p>

- [InstructKGC-CCKS2023 Evaluation of Instruction-based Knowledge Graph Construction](#instructkgc-ccks2023-evaluation-of-instruction-based-knowledge-graph-construction)
  - [🎯 1.Task Object](#-1task-object)
  - [📊 2.Data](#-2data)
    - [2.1Information Extraction Template](#21information-extraction-template)
    - [2.2Datasets](#22datasets)
    - [2.3Data Preprocessing](#23data-preprocessing)
  - [🚴 3.Preparation](#-3preparation)
    - [🛠️ 3.1Environment](#️-31environment)
    - [⏬ 3.2Download data](#-32download-data)
    - [🐐 3.3Model](#-33model)
  - [🌰 4.LoRA Fine-tuning](#-4lora-fine-tuning)
    - [4.1 Basic Parameters](#41-basic-parameters)
    - [4.2LoRA Fine-tuning with LLaMA](#42lora-fine-tuning-with-llama)
    - [4.3LoRA Fine-tuning with Alpaca](#43lora-fine-tuning-with-alpaca)
    - [4.4LoRA Fine-tuning with ZhiXi (智析)](#44lora-fine-tuning-with-zhixi-智析)
    - [4.5LoRA Fine-Tuning Vicuna](#45lora-fine-tuning-vicuna)
    - [4.6Lora Fine-tuning with ChatGLM](#46lora-fine-tuning-with-chatglm)
    - [4.7Lora Fine-tuning with Moss](#47lora-fine-tuning-with-moss)
    - [4.8LoRA Fine-tuning with Baichuan](#48lora-fine-tuning-with-baichuan)
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


## 🎯 1.Task Object

The task objective is to extract specified types of entities and relationships from a given text based on user-provided instructions, for the purpose of constructing a knowledge graph.

Here is an example of a **Knowledge Graph Construction Task**. The user provides a piece of text, referred to as the input, and an instruction that includes the desired types of entities or relationships to be extracted. The system's task is to output all the relationship triples contained in the input and return them in the format specified in the instruction (in this case, in the format of (head entity, relation, tail entity)).


```
instruction="You are an expert specifically trained in extracting relation triples. Given the candidate relation list: ['achievement', 'alternative name', 'area', 'creation time', 'creator', 'event', 'height', 'length', 'located in', 'named after', 'width'], please extract the possible head and tail entities from the input below based on the relation list, and provide the corresponding relation triple. If a certain relation does not exist, output NAN. Please answer in the format of (Subject,Relation,Object)\n."
input="Wewak Airport, also known as Boram Airport, is an airport located in Wewak, Papua New Guinea. IATA code: WWK; ICAO code: AYWK."
output="(Wewak Airport,located in,Wewak)\n(Wewak,located in,Papua New Guinea)\n(Wewak Airport,alternative name,Boram Airport)\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN\nNAN", "cate": "Building"
```


## 📊 2.Data


### 2.1Information Extraction Template
The `template` is used to construct an `instruction` for the input of model, consisting of three parts:
1. **Task Description**: Clearly define the model's function and the task it needs to complete, such as entity recognition, relation extraction, event extraction, etc.
2. **Candidate Label List {s_schema} (optional)**: Define the categories of labels that the model needs to extract, such as entity types, relation types, event types, etc.
3. **Structured Output Format {s_format}**: Specify how the model should present the structured information it extracts.


Template **with specified list of candidate labels**:
```
Named Entity Recognition(NER): You are an expert specialized in entity extraction. With the candidate entity types list: {s_schema}, please extract possible entities from the input below, outputting NAN if a certain entity does not exist. Respond in the format {s_format}.

Relation Extraction(RE): You are an expert in extracting relation triples. With the candidate relation list: {s_schema}, please extract the possible head entities and tail entities from the input below and provide the corresponding relation triples. If a relation does not exist, output NAN. Please answer in the {s_format} format.

Event Extraction(EE): You are a specialist in event extraction. Given the candidate event dictionary: {s_schema}, please extract any possible events from the input below. If an event does not exist, output NAN. Please answer in the format of {s_format}.

Event Type Extraction(EET): As an event analysis specialist, you need to review the input and determine possible events based on the event type directory: {s_schema}. All answers should be based on the {s_format} format. If the event type does not match, please mark with NAN.

Event Argument Extraction(EEA): You are an expert in event argument extraction. Given the event dictionary: {s_schema1}, and the event type and trigger words: {s_schema2}, please extract possible arguments from the following input. If an event argument does not exist, output NAN. Please respond in the {s_format} format.
```


<details>
    <summary><b>Template without specifying a list of candidate labels</b></summary>


  ```
  Named Entity Recognition(NER): Analyze the text content and extract the clear entities. Present your findings in the {s_format} format, skipping any ambiguous or uncertain parts.

  Relation Extraction(RE): Please extract all the relation triples from the text and present the results in the format of {s_format}. Ignore those entities that do not conform to the standard relation template.

  Event Extraction(EE): Please analyze the following text, extract all identifiable events, and present them in the specified format {s_format}. If certain information does not constitute an event, simply skip it.

  Event Type Extraction(EET): Examine the following text content and extract any events you deem significant. Provide your findings in the {s_format} format.

  Event Argument Extraction(EEA): Please extract possible arguments based on the event type and trigger word {s_schema2} from the input below. Answer in the format of {s_format}.
  ```
</details>


<details>
    <summary><b>Candidate Labels {s_schema}</b></summary>


    ```
    NER(Ontonotes): ["date", "organization", "person", "geographical social political", "national religious political", "facility", "cardinal", "location", "work of art", ...]
    RE(NYT): ["ethnicity", "place lived", "geographic distribution", "company industry", "country of administrative divisions", "administrative division of country", ...]
    EE(ACE2005): {"declare bankruptcy": ["organization"], "transfer ownership": ["artifact", "place", "seller", "buyer", "beneficiary"], "marry": ["person", "place"], ...}
    EET(GENIA): ["cell type", "cell line", "protein", "RNA", "DNA"]
    EEA(ACE2005): {"declare bankruptcy": ["organization"], "transfer ownership": ["artifact", "place", "seller", "buyer", "beneficiary"], "marry": ["person", "place"], ...}
    ```
</details>

Here [schema](./kg2instruction/convert/utils.py) provides 12 **text topics** and common relationship types under the topic.

<details>
    <summary><b>Structural Output Format {s_format}</b></summary>


    ```
    Named Entity Recognition(NER): (Entity,Entity Type)

    Relation Extraction(RE): (Subject,Relation,Object)

    Event Extraction(EE): (Event Trigger,Event Type,Argument1#Argument Role1;Argument2#Argument Role2)

    Event Type Extraction(EET): (Event Trigger,Event Type)

    Event Argument Extraction(EEA): (Event Trigger,Event Type,Argument1#Argument Role1;Argument2#Argument Role2)
    ```

</details>


For a more comprehensive understanding of the templates, please refer to the files [ner_converter.py](./kg2instruction/convert/converter/ner_converter.py)、[re_converter.py](./kg2instruction/convert/converter/re_converter.py)、[ee_converter.py](./kg2instruction/convert/converter/ee_converter.py)、[eet_converter.py](./kg2instruction/convert/converter/eet_converter.py)、[eea_converter.py](./kg2instruction/convert/converter/eea_converter.py) and [configs](./configs).




### 2.2Datasets

| Name                   | Download                                                     | Quantity | Description                                                  |
| ---------------------- | ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| InstructIE-train          | [Google drive](https://drive.google.com/file/d/1VX5buWC9qVeVuudh_mhc_nC7IPPpGchQ/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/1xXVrjkinw4cyKKFBR8BwQw?pwd=x4s7)  | 30w+  | InstructIE train set |
| InstructIE-valid       | [Google drive](https://drive.google.com/file/d/1EMvqYnnniKCGEYMLoENE1VD6DrcQ1Hhj/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE) <br/> [Baidu Netdisk](https://pan.baidu.com/s/11u_f_JT30W6B5xmUPC3enw?pwd=71ie)     | 2000+ | InstructIE validation set                                                                                     |
| InstructIE-test       | [Google drive](https://drive.google.com/file/d/1WdG6_ouS-dBjWUXLuROx03hP-1_QY5n4/view?usp=drive_link) <br/> [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowLM-IE)  <br/> [Baidu Netdisk](https://pan.baidu.com/s/1JiRiOoyBVOold58zY482TA?pwd=cyr9)     | 2000+ | InstructIE test set                                                                                    |
| train.json, valid.json | [Google drive](https://drive.google.com/file/d/1vfD4xgToVbCrFP2q-SD7iuRT2KWubIv9/view?usp=sharing) | 5,000    | Preliminary training set and test set for the task "Instruction-Driven Adaptive Knowledge Graph Construction" in [CCKS2023 Open Knowledge Graph Challenge](https://tianchi.aliyun.com/competition/entrance/532080/introduction), randomly selected from instruct_train.json |


The `InstructIE-train` dataset contains two core files: `InstructIE-zh.json` and `InstructIE-en.json`. Both files cover a range of fields that provide detailed descriptions of different aspects of the dataset:

- `'id'`: A unique identifier for each data entry, ensuring the independence and traceability of the data items.
- `'cate'`: The text's subject category, which provides a high-level categorical label for the content (there are 12 categories in total).
- `'entity'` and `'relation'`: Represent **entity and relationship triples**, respectively. These fields allow users to freely construct instructions and expected outputs for information extraction.

For the validation set `InstructIE-valid` and the test set `InstructIE-test`, they include **both Chinese and English versions**, ensuring the dataset's applicability in different language settings.

- `train.json`: The field definitions in this file are consistent with `InstructIE-train`, but the `'instruction'` and `'output'` fields show one format. Nonetheless, users can still freely construct instructions and outputs for information extraction based on the `'relation'` field.
- `valid.json`: Its field meanings are consistent with `train.json`, but this dataset has been **crowdsource-annotated**, providing higher accuracy and reliability.


<details>
  <summary><b>Explanation of each field</b></summary>


| Field       | Description                                                      |
| ----------- | ---------------------------------------------------------------- |
| id          | The unique identifier for each data point.                       |
| cate        | The category of the text's subject, with a total of 12 different thematic categories. |
| input       | The input text for the model, with the goal of extracting all the involved relationship triples. |
| instruction | Instructions guiding the model to perform information extraction tasks. |
| output      | The expected output result of the model.                         |
| entity      | Details describing the entity and its corresponding type (entity, entity_type). |
| relation    | Describes the relationship triples contained in the text, i.e., the connections between entities (head, relation, tail). |

</details>

With the fields mentioned above, users can flexibly design and implement instructions and output formats for different information extraction needs.


<details>
  <summary><b>Example of data</b></summary>


    ```
    {
        "id": "6e4f87f7f92b1b9bd5cb3d2c3f2cbbc364caaed30940a1f8b7b48b04e64ec403", 
        "cate": "Person", 
        "input": "Dionisio Pérez Gutiérrez  (born 1872 in Grazalema (Cádiz) - died 23 February 1935 in Madrid) was a Spanish writer, journalist, and gastronome. He has been called \"one of Spain's most authoritative food writers\" and was an early adopter of the term Hispanidad.\nHis pen name, \"Post-Thebussem\", was chosen as a show of support for Mariano Pardo de Figueroa, who went by the handle \"Dr. Thebussem\".", 
        "entity": [
            {"entity": "Dionisio Pérez Gutiérrez", "entity_type": "human"}, 
            {"entity": "Post-Thebussem", "entity_type": "human"}, 
            {"entity": "Grazalema", "entity_type": "geographic_region"}, 
            {"entity": "Cádiz", "entity_type": "geographic_region"}, 
            {"entity": "Madrid", "entity_type": "geographic_region"}, 
            {"entity": "gastronome", "entity_type": "event"}, 
            {"entity": "Spain", "entity_type": "geographic_region"}, 
            {"entity": "Hispanidad", "entity_type": "architectural_structure"}, 
            {"entity": "Mariano Pardo de Figueroa", "entity_type": "human"}, 
            {"entity": "23 February 1935", "entity_type": "time"}
        ], 
        "relation": [
            {"head": "Dionisio Pérez Gutiérrez", "relation": "country of citizenship", "tail": "Spain"}, 
            {"head": "Dionisio Pérez Gutiérrez", "relation": "place of birth", "tail":"Grazalema"}, 
            {"head": "Dionisio Pérez Gutiérrez", "relation": "place of death", "tail": "Madrid"}, 
            {"head": "Mariano Pardo de Figueroa", "relation": "country of citizenship", "tail": "Spain"}, 
            {"head": "Dionisio Pérez Gutiérrez", "relation": "alternative name", "tail": "Post-Thebussem"}, 
            {"head": "Dionisio Pérez Gutiérrez", "relation": "date of death", "tail": "23 February 1935"}
        ]
    }
    ```

</details>




### 2.3Data Preprocessing

**Training Data Transformation**

Before inputting data into the model, it needs to be formatted to include `instruction` and `input` fields. To assist with this, we offer a script [kg2instruction/convert.py](./kg2instruction/convert.py), which can batch convert data into a format directly usable by the model.

> Before using the [kg2instruction/convert.py](./kg2instruction/convert.py) script, please ensure you have referred to the [data](./data) directory. This directory lists in detail the data format requirements for each task.


```bash              
python kg2instruction/convert.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \      # Specifies the language for the conversion script and template, options are ['zh', 'en']
  --task NER \         # Specifies the task type: one of ['RE', 'NER', 'EE', 'EET', 'EEA']
  --sample -1 \        # If -1, randomly samples one of 20 instruction and 4 output formats; if a specific number, uses the corresponding instruction format, range is -1<=sample<20
  --neg_ratio 1 \      # Set the negative sampling ratio for all samples; 1 indicates negative sampling for all samples.
  --neg_schema 1 \     # Set the negative sampling ratio from the schema; 1 indicates embedding the entire schema into the command.
  --random_sort        # Determines whether to randomly sort the list of schemas in the instruction
```

**Negative Sampling**: Assuming dataset A contains labels [a, b, c, d, e, f], for a given sample s, it might involve only labels a and b. Our objective is to randomly introduce some relationships from the candidate relationship list that were originally unrelated to s, such as c and d. However, it's worth noting that in the output, the labels for c and d either won't be included, or they will be output as `NAN`.

`schema_path` is used to specify a schema file (in JSON format) containing three lines of JSON strings. Each line is organized in a fixed format and provides information for named entity recognition (NER) tasks. Taking NER tasks as an example, the meaning of each line is explained as follows:

```
["BookTitle", "Address", "Movie", ...]  # List of entity types
[]  # Empty list
{}  # Empty dictionary
```


<details>
  <summary><b>More</b></summary>



```
For Relation Extraction (RE) tasks:
[]                                                 # Empty list
["Founder", "Number", "RegisteredCapital", ...]    # List of relation types
{}                                                 # Empty dictionary


For Event Extraction (EE) tasks:
["Social Interaction-Thanks", "Organizational Action-OpeningCeremony", "Competition Action-Withdrawal", ...]        # List of event types
["DismissingParty", "TerminatingParty", "Reporter", "ArrestedPerson"]       # List of argument roles
{"OrganizationalRelation-Layoff": ["LayoffParty", "NumberLaidOff", "Time"], "LegalAction-Sue": ["Plaintiff", "Defendant", "Time"], ...}         # Dictionary of event types


For Event Type Extraction(EET) tasks:
["Social Interaction-Thanks", "Organizational Action-OpeningCeremony", "Competition Action-Withdrawal", ...]         # List of event types
[]                               # Empty list
{}                               # Empty dictionary


For Event Argument Extraction(EEA) tasks:
["Social Interaction-Thanks", "Organizational Action-OpeningCeremony", "Competition Action-Withdrawal", ...]                  # List of event types
["DismissingParty", "TerminatingParty", "Reporter", "ArrestedPerson"]           # List of argument roles
{"OrganizationalRelation-Layoff": ["LayoffParty", "NumberLaidOff", "Time"], "LegalAction-Sue": ["Plaintiff", "Defendant", "Time"], ...}             # Dictionary of event types
```

</details>

For more detailed information on the schema file, you can refer to the `schema.json` file in the respective task directories under the [data](./data) directory.


**Testing Data Transformation**

For test data, you can use the [kg2instruction/convert_test.py](./kg2instruction/convert_test.py) script, which does not require the data to contain label fields (`entity`, `relation`, `event`), just the input field and the corresponding schema_path.

```bash
python kg2instruction/convert_test.py \
  --src_path data/NER/sample.json \
  --tgt_path data/NER/processed.json \
  --schema_path data/NER/schema.json \
  --language zh \
  --task NER \
  --sample 0
```


**Data Transformation Examples**

Here is an example of data conversion for Named Entity Recognition (NER) task:

```
Before Transformation:
{
    "input": "In contrast, the rain-soaked battle between Qingdao Sea Bulls and Guangzhou Songri Team, although also ended in a 0:0 draw, was uneventful.",
    "entity": [{"entity": "Guangzhou Songri Team", "entity_type": "Organizational Structure"}, {"entity": "Qingdao Sea Bulls", "entity_type": "Organizational Structure"}]
}

After Transformation:
{
    "id": "e88d2b42f8ca14af1b77474fcb18671ed3cacc0c75cf91f63375e966574bd187",
    "instruction": "Please identify and list the entity types mentioned in the given text ['Organizational Structure', 'Person', 'Geographical Location']. If a type doesn't exist, please indicate it as NAN. Provide your answer in the format (entity, entity type).",
    "input": "In contrast, the rain-soaked battle between Qingdao Sea Bulls and Guangzhou Songri Team, although also ended in a 0:0 draw, was uneventful.",
    "output": "(Qingdao Sea Bulls,Organizational Structure)\n(Guangzhou Songri Team,Organizational Structure)\nNAN\nNAN"
}
```

Before conversion, the data format needs to adhere to the structure specified in the `DeepKE/example/llm/InstructKGC/data` directory for each task (such as NER, RE, EE). Taking NER task as an example, the input text should be labeled as the `input` field, and the annotated data should be labeled as the `entity` field, which is a list of dictionaries containing multiple key-value pairs for `entity` and `entity_type`.

After data conversion, you will obtain structured data containing the `input` text, `instruction` (providing detailed instructions about candidate entity types ['Organization', 'Person', 'Location'] and the expected output format), and `output` (listing all entity information recognized in the `input` in the form of (entity, entity type)).




## 🚴 3.Preparation


### 🛠️ 3.1Environment
Please refer to [DeepKE/example/llm/README.md](../README.md/#requirements) to create a Python virtual environment, and activate the `deepke-llm` environment:

```bash
conda activate deepke-llm
```

!!! Attention: To accommodate the `qlora` technique, we have upgraded the versions of the `transformers`, `accelerate`, `bitsandbytes`, and `peft` libraries in the original deepke-llm codebase.

1. transformers 0.17.1 -> 4.30.2
2. accelerate 4.28.1 -> 0.20.3
3. bitsandbytes 0.37.2 -> 0.39.1
4. peft 0.2.0 -> 0.4.0dev


### ⏬ 3.2Download data

```bash
mkdir results
mkdir lora
mkdir data
```

Place the data in the directory `./data`


### 🐐 3.3Model 
Here are some models:
* [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf) | [LLaMA-13b](https://huggingface.co/decapoda-research/llama-13b-hf)
* [zjunlp/knowlm-13b-base-v1.0](https://huggingface.co/zjunlp/knowlm-13b-base-v1.0)(需搭配相应的IE Lora) | [zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi)(无需Lora即可直接预测) | [zjunlp/knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie)(无需Lora, IE能力更强, 但通用性有所削弱)
* [baichuan-inc/Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B) | [baichuan-inc/Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) | [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) | [baichuan-inc/Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)


<details>
  <summary><b>more</b></summary>


* [Alpaca-7b](https://huggingface.co/circulus/alpaca-7b) | [Alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)
* [Vicuna-7b-delta-v1.1](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) | [Vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) | 
* [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
* [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
* [Chinese-LLaMA-7B](https://huggingface.co/Linly-AI/Chinese-LLaMA-7B)
</details>




## 🌰 4.LoRA Fine-tuning


### 4.1 Basic Parameters
When performing LoRA fine-tuning, you need to configure some basic parameters to specify the model type, dataset path, output settings, etc. Below are the available basic parameters and their descriptions:

* `--model_name`: Specifies the model name you wish to use. The current list of supported models includes: ["llama", "falcon", "baichuan", "chatglm", "moss", "alpaca", "vicuna", "zhixi"]. Note that this parameter should be distinguished from model_name_or_path.
* `--train_file` and `--valid_file` (optional): Point to the paths of your training and validation set JSON files, respectively. If a valid_file is not provided, the system will by default carve out a number of samples specified by val_set_size from the file indicated by train_file to be used as the validation set. You can also adjust the val_set_size parameter to change the number of samples in the validation set.
* `--output_dir`: Sets the path for saving the weight parameters after LoRA fine-tuning.
* `--val_set_size`: Defines the number of samples in the validation set, with a default of 1000.
* `--prompt_template_name`: Choose the name of the template you want to use. Currently, three types of templates are supported: [alpaca, vicuna, moss], with the alpaca template being the default.
* `--max_memory_MB` (default setting is 80000) is used to specify the size of the GPU memory. Please adjust it according to the performance of your GPU.
* For more information on parameter configuration, please refer to the [src/utils/args.py](./src/utils/args.py) file.


> Important Note: All the following commands should be executed in the InstrctKGC directory. For example, if you want to run a fine-tuning script, you should use the following command: bash scripts/fine_llama.bash. Make sure your current working directory is correct.


### 4.2LoRA Fine-tuning with LLaMA

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
    --bits 4 \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err
```

1. For the Llama model, we use [LLaMA-7b](https://huggingface.co/decapoda-research/llama-7b-hf).
2. For `prompt_template_name`, we use the alpaca template by default. The detailed contents of the template can be found in the [templates/alpaca.json](./templates/alpaca.json) file.
3. We have successfully run the finetuning code for the LLAMA model using LoRA technology on an RTX3090 GPU.
4. `model_name` = llama (llama2 is also llama).
  

The specific script for fine-tuning the LLAMA model can be found in [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash).



### 4.3LoRA Fine-tuning with Alpaca

When fine-tuning the Alpaca model, you can follow steps similar to those for [fine-tuning the LLaMA model](./README.md/#42lora-fine-tuning-with-llama). To fine-tune, make the following changes to the [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash) file:


```bash
output_dir='path to save Alpaca Lora'
--model_name_or_path 'path or name to Alpaca' \
--model_name 'alpaca' \
```

1. For the Alpaca model, we use [Alpaca-7b](https://huggingface.co/circulus/alpaca-7b).
2. For `prompt_template_name`, we default to using the alpaca template. The detailed contents of the template can be found in the [templates/alpaca.json](./templates/alpaca.json) file.
3. We have successfully run the finetuning code for the Alpaca model using LoRA technology on an RTX3090 GPU.
4. `model_name` = alpaca




### 4.4LoRA Fine-tuning with ZhiXi (智析)
Before starting to fine-tune the Zhixi model, ensure you follow the guide on [acquiring and restoring KnowLM2.2 pre-trained model weights](https://github.com/zjunlp/KnowLM#2-2) to obtain the complete Zhixi model weights.

**Important Note**: As the Zhixi model has already been trained on a rich set of information extraction task datasets using LoRA, you might not need to fine-tune it again and can proceed directly to prediction tasks. If you choose to conduct further training, follow the steps below.

The instructions for fine-tuning the Zhixi model are similar to those for [fine-tuning the LLaMA model](./README.md/#42lora-fine-tuning-with-llama), with the following adjustments in [ft_scripts/fine_llama.bash](./ft_scripts/fine_llama.bash):



```bash
output_dir='path to save Zhixi Lora'
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--model_name_or_path 'path or name to Zhixi' \
--model_name 'zhixi' \
```

1. Since Zhixi currently only has a 13b model, it is recommended to accordingly reduce the batch size.
2. For `prompt_template_name`, we default to using the alpaca template. The detailed contents of the template can be found in the [templates/alpaca.json](./templates/alpaca.json) file.
3. We have successfully run the fine-tuning code for the Zhixi model using LoRA technology on an RTX3090 GPU.
4. `model_name` = zhixi


### 4.5LoRA Fine-Tuning Vicuna

You can set your own parameters to fine-tune the Vicuna model using the LoRA method with the following commands:

<details>
  <summary><b>details</b></summary>


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
</details>

1. For the Vicuna model, we use [Vicuna-7b-delta-v1.1](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1)
2. Since the Vicuna-7b-delta-v1.1 uses a different `prompt_template_name` than the `alpaca` template, you need to set `--prompt_template_name 'vicuna'`, see [templates/vicuna.json](./templates//vicuna.json) for details
3. We have successfully run the vicuna-lora fine-tuning code on an `RTX3090`
4. `model_name` = vicuna

The corresponding script can be found at [ft_scripts/fine_vicuna.bash](./ft_scripts//fine_vicuna.bash)




### 4.6Lora Fine-tuning with ChatGLM

You can use the following command to configure your own parameters and fine-tune the ChatGLM model using the LoRA method:

<details>
  <summary><b>details</b></summary>


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
</details>

1. We use the [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) model for ChatGLM.
2. We use the default `alpaca` template for the `prompt_template_name`. Please refer to [templates/alpaca.json](./templates/alpaca.json) for more details.
3. Due to unsatisfactory performance with 8-bit quantization, we did not apply quantization to the ChatGLM model.
4. We have successfully run the ChatGLM-LoRA fine-tuning code on an `RTX3090` GPU.
5. model_name = chatglm

The corresponding script can be found at [ft_scripts/fine_chatglm.bash](./ft_scripts//fine_chatglm.bash).




### 4.7Lora Fine-tuning with Moss

You can use the following command to configure your own parameters and fine-tune the Moss model using the LoRA method:

<details>
  <summary><b>details</b></summary>


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
</details>

1. We use the [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft) model for Moss.
2. The `prompt_template_name` has been modified based on the alpaca template. Please refer to [templates/moss.json](./templates/moss.json) for more details. Therefore, you need to set `--prompt_template_name 'moss'`.
3. Due to memory limitations on the `RTX3090`, we use the `qlora` technique for 4-bit quantization. However, you can try 8-bit quantization or non-quantization strategies on `V100` or `A100` GPUs.
4. We have successfully run the Moss-LoRA fine-tuning code on an `RTX3090` GPU.
5. model_name = moss

The corresponding script can be found at [ft_scripts/fine_moss.bash](./ft_scripts/fine_moss.bash).



### 4.8LoRA Fine-tuning with Baichuan

You can use the following command to configure your own parameters and fine-tune the Llama model using the LoRA method:

<details>
  <summary><b>details</b></summary>


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
</details>

1. We use the [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) model for Llama.
2. There are currently some issues with evaluation, so we use `evaluation_strategy` 'no'.
3. We use the default `alpaca` template for the `prompt_template_name`. Please refer to [templates/alpaca.json](./templates/alpaca.json) for more details.
4. We have successfully run the Llama-LoRA fine-tuning code on an `RTX3090` GPU.
5. model_name = baichuan



The corresponding script can be found at [ft_scripts/fine_baichuan.bash](./ft_scripts/fine_baichuan.bash).




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
The following are some models that have been optimized through LoRA technique training (LoRA weights):
* [alpaca-7b-lora-ie](https://huggingface.co/zjunlp/alpaca-7b-lora-ie)
* [llama-7b-lora-ie](https://huggingface.co/zjunlp/llama-7b-lora-ie)
* [alpaca-13b-lora-ie](https://huggingface.co/zjunlp/alpaca-13b-lora-ie)
* [knowlm-13b-ie-lora](https://huggingface.co/zjunlp/knowlm-13b-ie-lora)

The following table shows the relationship between the base models and their corresponding LoRA weights:

| Base Model                 | LoRA Weights           |
| -------------------------- | ---------------------- |
| llama-7b                   | llama-7b-lora-ie       |
| alpaca-7b                  | alpaca-7b-lora-ie      |
| zjunlp/knowlm-13b-base-v1.0 | knowlm-13b-ie-lora         |


To use these trained LoRA models for prediction, you can execute the following command:

```bash
CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'model path or name' \
    --model_name 'model name' \
    --lora_weights 'path to LoRA weights' \
    --input_file 'data/valid.json' \
    --output_file 'results/results_valid.json' \
    --fp16 \
    --bits 4 
```


**Note**: Please ensure that the settings for `--fp16` or `--bf16`, `--bits`, `--prompt_template_name`, `--model_name` are consistent with the settings during [4.LoRA Fine-Tuning](./README_CN.md/#4lora微调).


#### 6.1.2 IE-Specific Model
If you want to use a trained model (without LoRA or with LoRA integrated into the model parameters), you can execute the following command for prediction:

```bash
CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --model_name_or_path 'model path or name' \
    --model_name 'model name' \
    --input_file 'data/valid.json' \
    --output_file 'results/results_valid.json' \
    --fp16 \
    --bits 4 
```

The following models are applicable to the above prediction method:
[zjunlp/knowlm-13b-zhixi](https://huggingface.co/zjunlp/knowlm-13b-zhixi) | [zjunlp/knowlm-13b-ie](https://huggingface.co/zjunlp/knowlm-13b-ie)


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
We provide a script, [evaluate.py](./kg2instruction/evaluate.py), to convert the model's string outputs into lists and calculate the F1 score.

```bash
python kg2instruction/evaluate.py \
  --standard_path data/NER/processed.json \
  --submit_path data/NER/processed.json \
  --task NER \
  --language zh
```


## 👋 8.Acknowledgment

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
