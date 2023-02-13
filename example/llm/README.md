Part of the code in this project comes from [Promptify](https://github.com/promptslab/Promptify). Thank you very much for the Promptify team.

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md">简体中文</a> </b>
</p>

## Contents
- [Contents](#Contents)

- [NER, EE and RTE with Large Language Models](#ner-ee-and-rte-with-large-language-models)
  - [Requirements and Datasets](#requirements-datasets-and-configuration)
  - [Example](#example)

- [Relation Extraction with Large Language Models](#relation-extraction-with-large-language-models)
  - [Requirements and Datasets](#requirements-and-datasets)
  - [Prompts](#prompts)
  - [In-context Learning](#in-context-learning)
  - [Data Generation via LLMs](#data-generation-via-llms)


# NER EE and RTE with Large Language Models

## Requirements Datasets and Configuration
- Requirements
  
  OpenAI API (key) is utilized for language models (e.g. GPT-3).
    ```shell
    >> pip install openai
    >> pip install jinja2
    >> pip install hydra-core
    ```

- Datasets and configuration
  In `data` folder, the given json file is in the format required by the data.
  
  The `conf` folder stores the set parameters. The parameters required to call the GPT3 interface are passed in through the files in this folder. In the named entity recognition task, `text_input` parameter is the prediction text, `examples` are examples with few or zero samples, which can be empty, `domain` is the domain of the prediction text, which can be empty, and `label` is the entity label set, which can also be empty. In the event extraction task, `text_input` parameter is the prediction text, `examples` are examples with few or zero samples, which can be empty, and `domain` is the domain of the prediction text, which can also be empty. In the union extraction task, `text_input` parameter is the prediction text, `examples` are examples with few or zero samples, which can be empty, and `domain` is the domain of the prediction text, which can also be empty.




## Example
  |                           Task                           |          Input           |    Output    |     
  | :----------------------------------------------------------: | :------------------------: | :------------: |
  | NER |           Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday.         |       [{'E': 'Country', 'W': 'Japan'}, {'E': 'Country', 'W': 'Syria'}, {'E': 'Competition', 'W': 'Asian Cup'}, {'E': 'Competition', 'W': 'Group C championship'}]   |          
  | EE | In Baghdad, a cameraman died when an American tank fired on the Palestine Hotel. |    event_list: [ event_type: [arguments: [role: "cameraman", argument: "Baghdad"], [role: "American tank", argument: "Palestine Hotel"]] ]      |  
  | RTE |           The most common audits were about waste and recycling.          |[['audit', 'type', 'waste'], ['audit', 'type', 'recycling']]|     



# Relation Extraction with Large Language Models

## Requirements and Datasets
- Requirements
  
  OpenAI API (key) is utilized for language models (e.g. GPT-3).
    ```shell
    >> pip install openai
    ```
- Datasets
  - [TACRED](https://nlp.stanford.edu/projects/tacred/)
  - [TACREV](https://github.com/DFKI-NLP/tacrev)
  - [RE-TACRED](https://github.com/gstoica27/Re-TACRED)


## Prompts
![prompt](LLM.png)

## In-context Learning
To elicit comprehension of the relation extraction task from large language models (LLMs), in-context learning is applied by providing LLMs with demonstrations in prompts. As shown above, two kinds of prompts are designed: **TEXT PROMPT** only with essential elements for RE and **INSTRUCT PROMPT** with constructions related to relation extraction. Meanwhile, entity types as schemas can also be added to prompts for better performance.

Conduct in-context learning with k-shot demonstrations:

```shell
>> python gpt3ICL.py -h
    usage: gpt3ICL.py [-h] --api_key API_KEY --train_path TRAIN_PATH --test_path TEST_PATH --output_success OUTPUT_SUCCESS --output_nores OUTPUT_NORES --prompt {text,text_schema,instruct,instruct_schema} [--k K]

    optional arguments:
      -h, --help            show this help message and exit
      --api_key API_KEY, -ak API_KEY
      --train_path TRAIN_PATH, -tp TRAIN_PATH
                            The path of training / demonstration data.
      --test_path TEST_PATH, -ttp TEST_PATH
                            The path of test data.
      --output_success OUTPUT_SUCCESS, -os OUTPUT_SUCCESS
                            The output directory of successful ICL samples.
      --output_nores OUTPUT_NORES, -on OUTPUT_NORES
                            The output directory of failed ICL samples.
      --prompt {text,text_schema,instruct,instruct_schema}
      --k K                 k-shot demonstrations
```

## Data Generation via LLMs

To complement the scarcity of labeled RE data in few-shot settings, utilize specific prompts with descriptions of data forms to guide LLMs to generate more in-domain labeled data autonomously as shown in the picture above.

Obtain augmented data:
```shell
>> python gpt3DA.py -h
  usage: gpt3DA.py [-h] --api_key API_KEY --demo_path DEMO_PATH --output_dir OUTPUT_DIR --dataset {tacred,tacrev,retacred} [--k K]

  optional arguments:
    -h, --help            show this help message and exit
    --api_key API_KEY, -ak API_KEY
    --demo_path DEMO_PATH, -dp DEMO_PATH
                          The directory of demonstration data.
    --output_dir OUTPUT_DIR
                          The output directory of generated data.
    --dataset {tacred,tacrev,retacred}
    --k K                 k-shot demonstrations
```

