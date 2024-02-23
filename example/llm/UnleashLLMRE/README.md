# How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/UnleashLLMRE/README_CN.md">简体中文</a> </b>
</p>


## Contents
- [Relation Extraction with Large Language Models](#relation-extraction-with-large-language-models)
  - [Contents](#contents)
  - [Requirements and Datasets](#requirements-and-datasets)
  - [Prompts](#prompts)
  - [In-context Learning](#in-context-learning)
  - [Data Generation via LLMs](#data-generation-via-llms)


## Requirements and Datasets
- Requirements
  
  OpenAI API (key) is utilized for language models (e.g. GPT-3, GPT-3.5).
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



## Citation

```BibTeX
@inproceedings{xu-etal-2023-unleash,
    title = "How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?",
    author = "Xu, Xin  and
      Zhu, Yuqi  and
      Wang, Xiaohan  and
      Zhang, Ningyu",
    editor = "Sadat Moosavi, Nafise  and
      Gurevych, Iryna  and
      Hou, Yufang  and
      Kim, Gyuwan  and
      Kim, Young Jin  and
      Schuster, Tal  and
      Agrawal, Ameeta",
    booktitle = "Proceedings of The Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sustainlp-1.13",
    doi = "10.18653/v1/2023.sustainlp-1.13",
    pages = "190--200",
}
```

