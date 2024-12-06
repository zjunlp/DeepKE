✨ **_These two projects are our previous research_**: ✨

> [CodeKGC-Code Language Models for Knowledge Graph Construction](#-codekgc-code-language-models-for-knowledge-graph-construction)


 > [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?](#-how-to-unleash-the-power-of-large-language-models-for-few-shot-relation-extraction)


# 📌 CodeKGC-Code Language Models for Knowledge Graph Construction

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/CodeKGC/README_CN.md">简体中文</a> </b>
</p>


To better address Relational Triple Extraction (rte) task in Knowledge Graph Construction, we have designed code-style prompts to model the structure of  Relational Triple, and used Code-LLMs to generate more accurate predictions. The key step of code-style prompt construction is to transform (text, output triples) pairs into semantically equivalent program language written in Python.

<div align=center>
<img src="./CodeKGC/codekgc_figure.png" width="85%" height="75%" />
</div>

## Data and Configuration

- Data

  The example data of `conll04` dataset is stored in the `codekgc/data` folder. The entire data is available in [here](https://drive.google.com/drive/folders/1vVKJIUzK4hIipfdEGmS0CCoFmUmZwOQV?usp=share_link). Users can customize your own data, but it is necessary to follow the given data format.

- Configuration

  The `codekgc/config.json` file contains the set parameters. The parameters required to load files and call the openai models are passed in through this file.

  Descriptions of these parameters are as follows:

  - `schema_path` defines the file path of the schema prompt. Schema prompt contains the pre-defined Python classes including **Relation** class, **Entity** class, **Triple** class and **Extract** class.

    The data format of schema prompt is as follows:

    ```python
    from typing import List
    class Rel:
        def __init__(self, name: str):
            self.name = name
    class Work_for(Rel):
    ...
    class Entity:
        def __init__(self, name: str):
            self.name = name
    class person(Entity):
    ...
    class Triple:
        def __init__(self, head: Entity, relation: Rel, tail: Entity):
            self.head = head
            self.relation = relation
            self.tail = tail
    class Extract:
        def __init__(self, triples: List[Triple] = []):
            self.triples = triples
    ```

  - `ICL_path` defines the file path of in-context examples.

    The data format of ICL prompt is as follows:

    ```python
    """ In 1856 , the 28th President of the United States , Thomas Woodrow Wilson , was born in Staunton , Va . """
    extract = Extract([Triple(person('Thomas Woodrow Wilson'), Rel('Live in'), location('Staunton , Va')),])
    ...
    ```

  - `example_path` defines the file path of the test example in conll04 dataset.

  - `openai_key` is your api key of openai.

  - `engine`, `temperature`, `max_tokens`, `n`... are the parameters required to pass in to call the openai api.

## Run and Examples

Once the parameters are set, you can directly run the `codekgc.py`：

```shell
>> cd codekgc
>> python codekgc.py
```

Below are input and output examples for Relational Triple Extraction (rte) task using code-style prompts:

**Input**:

```python
from typing import List
class Rel:
...(schema prompt)

""" In 1856 , the 28th President..."""
extract = Extract([Triple(person('Thomas Woodrow Wilson'), Rel('Live in'), location('Staunton , Va')),])
...(in-context examples)

""" Boston University 's Michael D. Papagiannis said he believes the crater was created 100 million years ago when a 50-mile-wide meteorite slammed into the Earth . """
```

**Output**:

```python
extract = Extract([Triple(person('Michael D. Papagiannis'), Rel('Work for'), organization('Boston University')),])
```
## Citation
If you use the code, please cite the following paper:

```bibtex
@article{DBLP:journals/corr/abs-2304-09048,
  author       = {Zhen Bi and
                  Jing Chen and
                  Yinuo Jiang and
                  Feiyu Xiong and
                  Wei Guo and
                  Huajun Chen and
                  Ningyu Zhang},
  title        = {CodeKGC: Code Language Model for Generative Knowledge Graph Construction},
  journal      = {CoRR},
  volume       = {abs/2304.09048},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2304.09048},
  doi          = {10.48550/arXiv.2304.09048},
  eprinttype    = {arXiv},
  eprint       = {2304.09048},
  timestamp    = {Mon, 24 Apr 2023 15:03:18 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2304-09048.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

-----

# 📌 How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?


<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/UnleashLLMRE/README_CN.md">简体中文</a> </b>
</p>


## Contents
- [📌 CodeKGC-Code Language Models for Knowledge Graph Construction](#-codekgc-code-language-models-for-knowledge-graph-construction)
  - [Data and Configuration](#data-and-configuration)
  - [Run and Examples](#run-and-examples)
  - [Citation](#citation)
- [📌 How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?](#-how-to-unleash-the-power-of-large-language-models-for-few-shot-relation-extraction)
  - [Contents](#contents)
  - [Requirements and Datasets](#requirements-and-datasets)
  - [Prompts](#prompts)
  - [In-context Learning](#in-context-learning)
  - [Data Generation via LLMs](#data-generation-via-llms)
  - [Citation](#citation-1)


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
![prompt](UnleashLLMRE/LLM.png)

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

-----

✨ **_Thanks for your reading!_** ✨