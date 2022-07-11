# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/few-shot/README_CN.md">简体中文</a> </b>
</p>

## Requirements

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

## Model

<div align=center>
<img src="knowprompt-model.png" width="75%" height="75%" />
</div>

Few-shot relation extraction based on the WWW2020 paper”[KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction](https://arxiv.org/pdf/2104.07650.pdf)"

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/few-shot
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset

  - Download the dataset to this directory.

    ```bash
    wget 120.27.214.45/Data/re/few_shot/data.tar.gz
    tar -xzvf data.tar.gz
    ```

  - The dataset [SEMEVAL](https://semeval2.fbk.eu/semeval2.php?location=tasks#T11) is stored in `data`:
    - `rel2id.json`：Relation Labels / Answer words - ID

    - `test.txt`： Test set

    - `train.txt`: Training set

    - `val.txt`：Validation set
  - We also provide [data augmentation methods](https://github.com/zjunlp/DeepKE/tree/main/example/re/few-shot/DA) to effectively leverage limited annotated RE data.
- Training

  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.

  - Few-shot training on SEMEVAL

    ```bash
    python run.py
    ```

  - The trained model is stored in the current directory by default.

  - Start to train from last-trained model<br>

    modify `train_from_saved_model` in `.yaml` as the path of the last-trained model

  - Logs for training are stored in the current directory by default and the path can be configured by modifying `log_dir` in `.yaml`

- Prediction

  ```bash
  python predict.py
  ```

## Cite

If you use or extend our work, please cite the following paper:

```bibtex
@inproceedings{DBLP:conf/www/ChenZXDYTHSC22,
  author    = {Xiang Chen and
               Ningyu Zhang and
               Xin Xie and
               Shumin Deng and
               Yunzhi Yao and
               Chuanqi Tan and
               Fei Huang and
               Luo Si and
               Huajun Chen},
  editor    = {Fr{\'{e}}d{\'{e}}rique Laforest and
               Rapha{\"{e}}l Troncy and
               Elena Simperl and
               Deepak Agarwal and
               Aristides Gionis and
               Ivan Herman and
               Lionel M{\'{e}}dini},
  title     = {KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization
               for Relation Extraction},
  booktitle = {{WWW} '22: The {ACM} Web Conference 2022, Virtual Event, Lyon, France,
               April 25 - 29, 2022},
  pages     = {2778--2788},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3485447.3511998},
  doi       = {10.1145/3485447.3511998},
  timestamp = {Tue, 26 Apr 2022 16:02:09 +0200},
  biburl    = {https://dblp.org/rec/conf/www/ChenZXDYTHSC22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
