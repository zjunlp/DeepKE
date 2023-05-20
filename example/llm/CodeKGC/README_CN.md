<p align="left">
    <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/CodeKGC/README.md">English</a> | 简体中文</a> </b>
</p>

# CodeKGC-基于代码语言模型的知识图谱构建

为了更好地处理知识图谱构建中的关系三元组抽取（RTE）任务，我们设计了代码形式的提示建模关系三元组的结构，并使用代码语言模型（Code-LLM）生成更准确的预测。代码形式提示构建的关键步骤是将（文本，输出三元组）对转换成Python中的语义等价的程序语言。

<div align=center>
<img src="./codekgc_figure.png" width="85%" height="75%" />
</div>

## 数据与参数

- 数据

  `conll04`数据集的样例数据存储在`codekgc/data`文件夹中。完整的数据可以在[这里](https://drive.google.com/drive/folders/1vVKJIUzK4hIipfdEGmS0CCoFmUmZwOQV?usp=share_link)获取。用户可以定制自己的数据，但必须遵循给定的数据格式。

- 参数

  `codekgc/config.json` 文件包含了设置参数。加载文件和调用 openai 模型所需的参数通过此文件传递。

  以下是这些参数的描述：

  - `schema_path` 定义了schema提示文件的文件路径。schema提示包含了预定义的 Python 类，包括 **Relation** 类、**Entity** 类、**Triple** 类和 **Extract** 类。 

    schema提示的数据格式如下：

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

  - `ICL_path` 定义了上下文示例的文件路径。

    ICL 提示的数据格式如下：

    ```python
    """ In 1856 , the 28th President of the United States , Thomas Woodrow Wilson , was born in Staunton , Va . """
    extract = Extract([Triple(person('Thomas Woodrow Wilson'), Rel('Live in'), location('Staunton , Va')),])
    ...
    ```

  - `example_path`定义了 conll04 数据集中测试示例的文件路径。

  - `openai_key` 是用户的 OpenAI API 密钥。

  - `engine`、`temperature`、`max_tokens`、`n`... 是调用 OpenAI API 时需要传递的参数。

## 使用与示例

当参数设置完成时，可以直接运行 `codekgc.py` 文件：

```shell
>> cd codekgc
>> python codekgc.py
```

以下是使用代码形式提示进行关系三元组抽取（RTE）任务的输入和输出示例：

**输入**：

```python
from typing import List
class Rel:
...(schema prompt)

""" In 1856 , the 28th President..."""
extract = Extract([Triple(person('Thomas Woodrow Wilson'), Rel('Live in'), location('Staunton , Va')),])
...(in-context examples)

""" Boston University 's Michael D. Papagiannis said he believes the crater was created 100 million years ago when a 50-mile-wide meteorite slammed into the Earth . """
```

**输出**：

```python
extract = Extract([Triple(person('Michael D. Papagiannis'), Rel('Work for'), organization('Boston University')),])
```
## 引用
如果您使用该代码，请引用以下论文：
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
