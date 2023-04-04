# Easy Start



<b> English | <a href="./README_CN.md">简体中文</a>  </b>


## Model

<div align=center>
<img src="ASP.png" width="75%" height="75%" />
</div>

Illustration of **ASP** (EMNLP'22) for entity and relation extraction (Details in paper [Autoregressive Structured Prediction with Language Models](https://aclanthology.org/2022.findings-emnlp.70.pdf)).

## Requirements

> python == 3.8.16

- tqdm==4.64.1
- numpy==1.24.1
- scipy==1.10.1
- torch==1.13.1+cu116
- huggingface_hub==0.12.1
- truecase==0.0.14
- pyhocon==0.3.60
- sentencepiece==0.1.97
- wandb==0.13.9
- hydra-core==1.3.1
- transformers==4.26.0

## Download Code

```
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/triple/ASP
```

## Install with Pip or Conda

- Create a new environment with pip or conda

  1. pip

  ```
  python -m venv <path_to_venv>/asp  
  source <path_to_venv>/asp/bin/activate
  pip install -r requirements.txt
  ```

  2. conda

  ```
  conda create -n asp python=3.8.16 # create a new environment (asp)
  ```

- install dependencies

  1. download deepke	

     You need to change `numpy==1.18.5` to `numpy==1.24.1` in requirement.txt under the directory named `deepke`.

  ```
  cd ~/DeepKE
  python setup.py build
  python setup.py install
  ```

  2. install dependencies with pip

  ```
  pip install -r requirements.txt
  ```

  2. install apex

  ```
  cd ~/DeepKE/example/triple/ASP
  git clone https://github.com/NVIDIA/apex
  cd apex
  ```

  ​	You need to modify line 32 of the `setup.py` under the directory named `apex` as follows.

  ```
   if (bare_metal_version != torch_binary_version):
      	pass
      	#raise RuntimeError(
      	#   "Cuda extensions are being compiled with a version of Cuda that does "
      	#    "not match the version used to compile Pytorch binaries.  "
      	#    "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
      	#    + "In some cases, a minor-version mismatch will not cause later errors:  "
      	#    "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
      	#    "You can try commenting out this check (at your own risk)."
      	#)
  ```

  ​	Then

  ```
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

## Train and Evaluate

- Dataset

  - Download the dataset.

    ```
    cd ~/DeepKE/example/triple/ASP/data
    wget 120.27.214.45/Data/triple/ASP/CMeIE.zip
    unzip ./CMeIE.zip
    rm ./CMeIE.zip
    cd ..
    ```

    CMeIE dataset：

    - `train.json`: Training set
    - `dev.json `: Validation set
    - `test.json`: Test set

- Training

  - `python run_ere.py <config_name> <gpu_id>`，Parameters for training are in the `conf` folder and users can modify them before training.This task supports multi card training.`config_name`is the name of dataset，gpu_id  is the main card for calculation，which requires a little more memory.Then run as follows.

  ```
  export ASP=$PWD
  python run_ere.py CMeIE
  ```

- Logs for training are in the `data/CMeIE/CMeIE`folder and the trained model is saved in this folder too.

- Evaluation

  `CUDA_VISIBLE_DEVICES=0 python evaluate_ere.py <config_name> <saved_suffix> <gpu_id>`，`config_name`is the name of dataset，save_suffix is the model， which is saved in the `data/CMeIE/CMeIE` folder，gpu_id is the main card for calculation，which requires a little more memory.Then run as follows. Then run as follows.

  ```
  CUDA_VISIBLE_DEVICES=0 python evaluate_ere.py CMeIE Mar05_19-39-56_2000 0
  ```

## Models

1. ASP (Based on the paper ["Autoregressive Structured Prediction with Language Models"](https://arxiv.org/abs/2210.14698))

## Data Labeling

If you only have sentence and entity pairs but relation labels, you can get use our distant supervised based [relation labeling tools](https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data).

Please make sure that:

- Use the triple file we provide or high-quality customized triple file
- Enough source data

## Cite 

```bibtex
@inproceedings{DBLP:conf/emnlp/LiuJMCS22,
  author    = {Tianyu Liu and
               Yuchen Eleanor Jiang and
               Nicholas Monath and
               Ryan Cotterell and
               Mrinmaya Sachan},
  editor    = {Yoav Goldberg and
               Zornitsa Kozareva and
               Yue Zhang},
  title     = {Autoregressive Structured Prediction with Language Models},
  booktitle = {Findings of the Association for Computational Linguistics: {EMNLP}
               2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022},
  pages     = {993--1005},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.findings-emnlp.70},
  timestamp = {Tue, 07 Feb 2023 17:10:51 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/LiuJMCS22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
