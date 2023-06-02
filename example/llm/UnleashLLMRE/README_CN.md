# 如何释放大型语言模型进行少样本关系抽取中的能力?

<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/UnleashLLMRE/README.md">English</a> | 简体中文</a> </b>
</p>

## 目录
- [使用大型语言模型进行关系抽取](#使用大型语言模型进行关系抽取)
  - [目录](#目录)
  - [环境和数据集](#环境和数据集)
  - [提示](#提示)
  - [上下文学习](#上下文学习)
  - [使用大型语言模型生成数据](#使用大型语言模型生成数据)


## 环境和数据集
- 环境配置
  
  使用GPT-3.5需要OpenAI API (key) 
    ```shell
    >> pip install openai
    ```
- 数据集
  - [TACRED](https://nlp.stanford.edu/projects/tacred/)
  - [TACREV](https://github.com/DFKI-NLP/tacrev)
  - [RE-TACRED](https://github.com/gstoica27/Re-TACRED)


## 提示
![prompt](LLM.png)

## 上下文学习
我们通过提供带有示例的提示给大型语言模型，使用上下文学习引导大型语言模型理解关系抽取任务。如上图所示，我们设计了两种提示：(1) **TEXT PROMPT** 仅包含关系抽取任务最基本的必要信息；(2) **INSTRUCT PROMPT** 包含关系抽取任务相关的指令。同时，为了实现更好的抽取性效果，实体类型可以作为schemas信息加入到提示中。

请使用以下脚本实现K-shot示例的上下文学习：

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

## 使用大型语言模型生成数据
为了弥补少样本场景下关系抽取有标签数据的缺失, 我们设计带有数据样式描述的提示，用于指导大型语言模型自动地生成更多的有标签数据根据已有的少样本数据，如上图最后一条提示所示。

请使用一下脚本生成数据:
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



## 引用

```BibTex
@article{UnleashLLMRE,
  author       = {Xin Xu and
                  Yuqi Zhu and
                  Xiaohan Wang and
                  Ningyu Zhang},
  title        = {How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?},
  journal      = {The 4th Workshop on Simple and Efficient Natural Language Processing (SustaiNLP 2023)},
  year         = {2023},
  url          = {https://arxiv.org/pdf/2305.01555.pdf},
  publisher    = "Association for Computational Linguistics"
}
```

