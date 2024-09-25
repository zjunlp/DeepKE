# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/cross/README_CN.md">简体中文</a> </b>
</p>

## 模型

<div align=center>
<img src="model.png" width="75%" height="75%" />
</div>

基于文本到文本生成模型的跨域命名实体识别方法**CP-NER** (详情请查看IJCAI2023论文[One Model for All Domains: Collaborative Domain-Prefix Tuning for Cross-Domain NER](https://arxiv.org/abs/2301.10410)).


## 环境依赖

> python == 3.8

- torch == 1.11
- transformers == 4.26.0
- datasets
- deepke

## 克隆代码

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/cross
```

## 使用pip安装

- 首先创建python虚拟环境，再进入虚拟环境
- 安装依赖: `pip install -r requirements.txt`.

## 数据集

  - 先使用以下代码下载处理好的数据：

    ```bash
    wget 120.27.214.45/Data/ner/cross/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - 解压后数据会存放到`data`文件夹中，包括了 CoNLL-2003, MIT-movie, MIT-restaurant, Ai, Literature, Music, Politics和science等数据.

  - 每个数据集都遵循以下的数据格式：
    - `train.json`：训练数据
    - `val.json `：验证数据
    - `test.json`：测试数据
    - `entity.schema`：实体类别
    - `event.schema`
    - `record.schema`
    - `relation.schema`

## 训练

1. 训练作为领域控制器的前缀 (Prefix)：

    - 我们首先利用领域的数据来预热训练对应的前缀。

    - 模型加载和训练相关的参数、路径等可以在 `hydra/run/train.yaml` 文件中进行修改。

      以 `CoNLL03` 的训练过程为例，先进行训练相关参数的修改：
      ```yaml
      train_file: 'data/conll03/train.json'
      validation_file: 'data/conll03/val.json'
      test_file: 'data/conll03/test.json'
      record_schema: '../../data/conll03/record.schema'
      output_dir: 'output/conll03-t5-base'        # 模型和训练数据的输出路径
      logging_dir: 'output/conll03-t5-base_log'   # 训练日志的路径
      model_name_or_path: '../../hf_models/t5-base' # 预训练模型的路径
      ```

      之后运行下列命令进行训练
      ```bash 
      python run.py
      ```
    - 训练过程中产生的最佳模型权重、训练细节和测试结果会存储到`logs/xxx/output/conll03-t5-base`.
    
    - 可以在此链接 [Google Drive](https://drive.google.com/file/d/1u7jg0AWzCB_dlExGG3RGgxJecc5A_3Rb/view?usp=sharing) 下载此步骤中产生的最佳模型权重。


2. 单领域迁移(Cross-NER)

    - 首先更改 `hydra/run/single_transfer.yaml` 中的参数 `model_name_or_path`, `source_prefix_path` 和 `targets_prefix_path`（通常 `model_path` 和 `targets_prefix_path` 相同的, `source_prefix_path` 是源域中训练好的模型路径)
    
    - 以 `CoNLL03` 迁移到 `AI` 为例，先进行相关参数的修改：
      ```yaml
      train_file: 'data/ai/train.json'  # AI领域的训练数据路径
      validation_file: 'data/ai/val.json'
      test_file: 'data/ai/test.json'
      record_schema: '../../data/ai/record.schema'
      output_dir: 'output/conll_to_ai-t5-base'            # 模型和训练数据的输出路径
      logging_dir: 'output/conll_to_ai-t5-base_log'       # 训练日志的路径
      model_name_or_path: '../xxx/output/ai-t5-base'      # 加载的模型路径
      source_prefix_path: '../xxx/output//conll-t5-base'  # 源域训好的模型路径
      target_prefix_path: '../xxx/output/ai-t5-base'      # 目标域训好的模型路径
      ```

    - 之后运行以下命令:
      ```bash
      python run.py hydra/run=single_transfer.yaml
      ```

3. 多源域迁移
    - 保存**每个域**的前缀和标签词信息。我们使用双查询域选择器（前缀和标签词）来聚合多个源域。 以`CoNLL03` 的保存过程为例，先进行相关参数的修改： 
      ```yaml
      output_dir: '../xxx/output/conll-t5-base'         # 前缀和标签词输出路径
      model_name_or_path: '../xxx/output/conll-t5-base' # 训好的模型的路径
      model_ckpt_path: '../xxx/output/conll-t5-base'    # 训好的模型的路径
      save_prefix: true     # 是否保存前缀
      save_label_word: true # 是否保存标签词
      ```
      之后运行以下命令，前缀和目标词会存储于`output_dir`文件夹下：
      ```bash
      python run.py hydra/run=save_prefix_label.yaml
      ```

    - 开始迁移。修改 `hydra/run/multi_transfer.yaml` 中的参数 `model_name_or_path`, `model_ckpt_path` 和 `multi_source_path`（注意 `multi_source_path` 包含多个源域的模型加载路径，他们用逗号进行分隔）。以 `CoNLL03`, `Politics`, `Music` 和 `Literature` 迁移到 `AI` 为例，先进行相关参数的修改：
      ```yaml
      model_name_or_path: '../xxx/output/ai-t5-base' # 目标域训好的模型路径
      model_ckpt_path: '../xxx/output/ai-t5-base'  # 目标域训好的模型路径
      multi_source_path: '../xxx/output/conll-t5-base,../xxx/output/politics-t5-base,../xxx/output/music-t5-base,../xxx/output/literature-t5-base' # 源域训好的模型路径 (用逗号进行分隔)
      ```
      最后运行以下命令进行迁移实验:
      ```bash
      python run.py hydra/run=multi_transfer.yaml
      ```


## 致谢
我们的代码基于 [UIE](https://github.com/universal-ie/UIE), 非常感谢！

## 引用

如果您使用了上述代码，请您引用下列论文:

```bibtex
@article{DBLP:journals/corr/abs-2301-10410,
  author    = {Xiang Chen and
               Lei Li and
               Shuofei Qiao and
               Ningyu Zhang and
               Chuanqi Tan and
               Yong Jiang and
               Fei Huang and
               Huajun Chen},
  title     = {One Model for All Domains: Collaborative Domain-Prefix Tuning for
               Cross-Domain {NER}},
  journal   = {CoRR},
  volume    = {abs/2301.10410},
  year      = {2023},
  url       = {https://doi.org/10.48550/arXiv.2301.10410},
  doi       = {10.48550/arXiv.2301.10410},
  eprinttype = {arXiv},
  eprint    = {2301.10410},
  timestamp = {Mon, 13 Mar 2023 11:20:37 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2301-10410.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
