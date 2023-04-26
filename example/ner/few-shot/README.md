# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/tree/main/example/triple/mt5/README_ZH.md">简体中文</a> </b>
</p>

## Model

<div align=center>
<img src="lightner-model.png" width="75%" height="75%" />
</div>

Illustration of **LightNER** (COLING'22) for few-shot named entity recognition (Details in paper [LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting](https://aclanthology.org/2022.coling-1.209.pdf)).
- ❗NOTE: We have released a follow-up work "[One Model for All Domains: Collaborative Domain-Prefix Tuning for Cross-Domain NER](https://arxiv.org/abs/2301.10410)" at [CP-NER](https://github.com/zjunlp/DeepKE/tree/main/example/ner/cross).

## Requirements

> python == 3.8

- torch == 1.11
- transformers == 4.26.0
- deepke

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/few-shot
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    ```bash
    wget 120.27.214.45/Data/ner/few_shot/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - The datasets are stored in `data`, including CoNLL-2003, MIT-movie, MIT-restaurant and ATIS.
    
  - **CoNLL-2003**
    
    - `train.txt`: Training set
    - `valid.txt `: Validation set
    - `test.txt`: Test set
    - `indomain-train.txt`: In-domain training set

  - **MIT-movie, MIT-restaurant and ATIS**
    - `k-shot-train.txt`: k=[10, 20, 50, 100, 200, 500], Training set
    - `test.txt`: Testing set

- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.

  - Training on CoNLL-2003

    ```bash
    python run.py
    ```

  - Few-shot Training

    If the model need to be uploaded, modify `load_path` in `few_shot.yaml`

    ```bash
    python run.py +train=few_shot
    ```

  - Few-shot Training for **Chinese**

    > Full data fine-tuning can achieve the best performance.

    Pretrained weights need to be provided in the directory defined in `few_shot_cn.yaml`
    ```bash
    python run.py +train=few_shot_cn
    ```

  - Logs for training are in the `log` folder. The path of the trained model can be customized.

- Prediction

  - Add `- predict` in `config.yaml`

  - Modify `load_path` as the path of the trained model and `write_path` as the path of predicted results in `predict.yaml` 

  - ```bash
    python predict.py
    ```

### Custom Tokenizer

If you need to customize your own Tokenizer (eg `MBartTokenizer` for multilingual processing).

You can customize the tokenizer in <a href="https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/few_shot/module/datasets.py#L18">tokenizer</a>

## Cite

If you use or extend our work, please cite the following paper:

```bibtex
@inproceedings{DBLP:conf/coling/00160DTXHSCZ22,
  author    = {Xiang Chen and
               Lei Li and
               Shumin Deng and
               Chuanqi Tan and
               Changliang Xu and
               Fei Huang and
               Luo Si and
               Huajun Chen and
               Ningyu Zhang},
  editor    = {Nicoletta Calzolari and
               Chu{-}Ren Huang and
               Hansaem Kim and
               James Pustejovsky and
               Leo Wanner and
               Key{-}Sun Choi and
               Pum{-}Mo Ryu and
               Hsin{-}Hsi Chen and
               Lucia Donatelli and
               Heng Ji and
               Sadao Kurohashi and
               Patrizia Paggio and
               Nianwen Xue and
               Seokhwan Kim and
               Younggyun Hahm and
               Zhong He and
               Tony Kyungil Lee and
               Enrico Santus and
               Francis Bond and
               Seung{-}Hoon Na},
  title     = {LightNER: {A} Lightweight Tuning Paradigm for Low-resource {NER} via
               Pluggable Prompting},
  booktitle = {Proceedings of the 29th International Conference on Computational
               Linguistics, {COLING} 2022, Gyeongju, Republic of Korea, October 12-17,
               2022},
  pages     = {2374--2387},
  publisher = {International Committee on Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.coling-1.209},
  timestamp = {Mon, 13 Mar 2023 11:20:33 +0100},
  biburl    = {https://dblp.org/rec/conf/coling/00160DTXHSCZ22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
