# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/README_CN.md">简体中文</a> </b>
</p>

### Model

This project implements extraction models for NER tasks in the **Standard** scenario. The corresponding paths are:  
* [BiLSTM-CRF](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/models/BiLSTM_CRF.py)  
* [Bert](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/models/InferBert.py) (Tip: If using the dataset provided below, we recommend setting learning_rate to 2e-5 and num_train_epochs to 10)
* [W2NER](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/w2ner)  

---

### Experimental Results

| Model        | Accuracy | Recall | F1 Score | Inference Speed ([People's Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily)) |
|--------------|----------|--------|----------|---------------------------------------------------------------------------------------------------------|
| BERT         | 91.15    | 93.68  | 92.40    | 106s                                                                                                    |
| BiLSTM-CRF   | 92.11    | 88.56  | 90.29    | 39s                                                                                                     |
| W2NER        | 96.76    | 96.11  | 96.43    | -                                                                                                       |

---

### Clone the Repository

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/standard
```

---

### Environment Setup

#### 1. Create a Python virtual environment and activate it:
   ```bash
   conda create -n deepke python=3.8
   conda activate deepke
   ```
#### 2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Parameter Configuration

#### 1. Model Parameters

Model-specific configurations (e.g., model path, hidden layer dimensions, case sensitivity) can be found in the [`conf/hydra/model/*.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf/hydra/model) directory.

#### 2. Other Parameters

Settings for environment paths and other hyperparameters during training are located in [`train.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf/train.yaml) and [`custom.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf/custom.yaml).

> **Note**: Vocabulary usage during training:
> - For the `Bert` model, the vocabulary is derived from the pre-trained weights on Hugging Face.
> - For `BiLSTM-CRF`, the vocabulary must be built based on the training dataset and saved in a `.pkl` file for prediction and evaluation (configured in the `model_vocab_path` attribute of `lstmcrf.yaml`).
> - For model downloads with network error, we recommend using the [Hugging Face mirror site](https://hf-mirror.com/) . After downloading, modify the model path in `conf/hydra/model/*.yaml`.
---

### Training with Dataset

#### 1. Supported Data Formats
   The model supports `json`, `docx`, and `txt` formats. For details, refer to the `data` folder. The default dataset is the **People's Daily** (Chinese NER) with text data in `{word, label}` pairs.  
   - **Note for English datasets**: Update `lan` in `config.yaml` before prediction, and install `nltk` with `nltk.download('punkt')`.

#### 2. Prepare Data  
   Download the dataset:
   ```bash
   wget 121.41.117.246/Data/ner/standard/data.tar.gz
   tar -xzvf data.tar.gz
   ```
   Place the following files in the `data` folder:
   - `train.txt`: Training dataset  
   - `valid.txt`: Validation dataset  
   - `test.txt`: Test dataset  

#### 3. Start Training
Choose the appropriate model for your target scenario:  
1. **Bert**  
   ```bash
   python run_bert.py
   ```   
   -  Update `hydra/model` in [config.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/conf/config.yaml) to `bert`. Hyperparameters for BERT are in [bert.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/conf/hydra/model/bert.yaml). 
   - Multi-GPU training is supported by setting `use_multi_gpu` to `True` in `train.yaml`. Specify GPUs with `os.environ['CUDA_VISIBLE_DEVICES']`.
   - For the default dataset downloaded above: Set `learning_rate` to `2e-5` and `num_train_epochs` to `10` in `train.yaml`.
   - Tip: If the dataset is small, reduce the learning rate to avoid fast parameter updates. For large datasets, try increasing the learning rate to speed up convergence. You can also monitor the training process using wandb (set config.yaml's `use_wandb` parameter to True) and adjust the learning rate and epochs for optimal performance.
2. **BiLSTM-CRF**  
   ```bash
   python run_lstmcrf.py
   ```  
     Configure BiLSTM-CRF hyperparameters in `lstmcrf.yaml`. Modify other training parameters in the `conf` folder.
3. **W2NER**  
   ```bash
   cd w2ner
   python run.py
   ```   
     Hyperparameters are in `model.yaml`. Specify the GPU index using the `device` parameter (set to 0 for single GPU setups).

#### Training Output
- **Logs and Results**  
   - Training logs are saved in the `logs` folder.  
   - Model checkpoints are stored in the `checkpoints` folder.
- **Batch Size**  
   - For BERT training, a batch size of 64 or more is recommended.

#### Prediction
   Run predictions using:
   ```bash
   python predict.py
   ```

## Prepare weak_supervised data

If you only have text data and corresponding dictionaries, but no canonical training data.

You can get weakly supervised formatted training data through automated labeling methods.

Please make sure that:

- Provide high-quality dictionaries
- Enough text data

<p align="left">
<a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data/README.md">prepare-data</a> </b>
</p>
