# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/few-shot/README_CN.md">简体中文</a> </b>
</p>

## Model

<div align=center>
<img src="lightner-model.png" width="75%" height="75%" />
</div>

Illustration of LightNER (COLING'22) for few-shot named entity recognition (Details in paper [https://arxiv.org/pdf/2109.00720.pdf](https://arxiv.org/pdf/2109.00720.pdf)).

## Requirements

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
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

  - Logs for training are in the `log` folder. The path of the trained model can be customized.

- Prediction

  - Add `- predict` in `config.yaml`

  - Modify `load_path` as the path of the trained model and `write_path` as the path of predicted results in `predict.yaml` 

  - ```bash
    python predict.py
    ```

## Cite

If you use or extend our work, please cite the following paper:

```bibtex
@article{DBLP:journals/corr/abs-2109-00720,
  author    = {Xiang Chen and
               Ningyu Zhang and
               Lei Li and
               Xin Xie and
               Shumin Deng and
               Chuanqi Tan and
               Fei Huang and
               Luo Si and
               Huajun Chen},
  title     = {LightNER: {A} Lightweight Generative Framework with Prompt-guided
               Attention for Low-resource {NER}},
  journal   = {CoRR},
  volume    = {abs/2109.00720},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.00720},
  eprinttype = {arXiv},
  eprint    = {2109.00720},
  timestamp = {Mon, 20 Sep 2021 16:29:41 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2109-00720.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
