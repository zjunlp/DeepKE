### 快速上手

<p align="left">
    <b><a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/README.md">English</a> | 简体中文</b>
</p>

---

### 模型内容

本项目实现了 **Standard** 场景下的 NER 任务提取模型，具体实现如下：
- [**BiLSTM-CRF**](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/models/BiLSTM_CRF.py)  
- [**Bert**](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/models/InferBert.py) （提示：若使用我们下文给出的数据集，推荐设置`learning_rate`为`2e-5`，`num_train_epochs`为`10`）  
- [**W2NER**](https://github.com/zjunlp/DeepKE/blob/main/src/deepke/name_entity_re/standard/w2ner)  

---

### 实验结果

| 模型         | 准确率   | 召回率   | F1 值  | 推理速度 ([People's Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily)) |
|--------------|----------|----------|--------|---------------------------------------------------------------------------------------------------------|
| **BERT**     | 91.15    | 93.68    | 92.40  | 106s                                                                                                    |
| **BiLSTM-CRF** | 92.11    | 88.56    | 90.29  | 39s                                                                                                     |
| **W2NER**    | 96.76    | 96.11    | 96.43  | -                                                                                                       |

---

### 克隆代码

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/standard
```

---

### 环境依赖

#### 1. 创建虚拟环境
```bash
conda create -n deepke python=3.8
conda activate deepke
```

#### 2. 安装依赖
```bash
pip install -r requirements.txt
```

---

### 参数设置

#### 1. 模型参数
模型的参数配置文件存放在 [`conf/hydra/model/*.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf/hydra/model) 路径下，例如模型路径、隐藏层维度、大小写敏感设置等。

#### 2. 其他参数
其他超参数（如环境路径、训练参数）存放于 [`train.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf/train.yaml) 和 [`custom.yaml`](https://github.com/zjunlp/DeepKE/tree/main/example/ner/standard/conf/custom.yaml) 中。

**注意**： 
- 使用 `Bert` 模型时，词典（vocab）来自 Hugging Face 的预训练权重。
- 使用 `BiLSTM-CRF` 模型时，需基于训练集构建词典，并保存为 `.pkl` 文件供预测和评估使用（配置在 `lstmcrf.yaml` 的 `model_vocab_path` 属性中）。
- 模型下载推荐使用[Hugging Face镜像网站](https://hf-mirror.com/) ，下载到本地后修改`conf/hydra/model/*.yaml`中的模型路径。

---

### 使用数据进行训练

#### 数据准备
- **支持格式**：`json`、`docx` 和 `txt`，详情请参考 `data` 文件夹。
- **默认数据**：本项目采用 **People's Daily** 中文 NER 数据集，格式为 `{word, label}` 对。  
- **英文数据集**：如需使用英文数据集，请在预测前修改 `config.yaml` 中的 `lan` 参数，并安装 `nltk` 和 `nltk.download('punkt')`。

#### 数据存放
下载数据并解压：
```bash
wget 120.27.214.45/Data/ner/standard/data.tar.gz
tar -xzvf data.tar.gz
```
将数据存放至 `data` 文件夹：
- `train.txt`：训练数据集  
- `valid.txt`：验证数据集  
- `test.txt`：测试数据集  

#### 开始训练
根据目标场景选择模型：
1. **Bert**  
   ```bash
   python run_bert.py
   ```  
   - 修改 [config.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/conf/config.yaml) 中 `hydra/model` 为 `bert`，超参数设置详见 [bert.yaml](https://github.com/zjunlp/DeepKE/blob/main/example/ner/standard/conf/hydra/model/bert.yaml)。
   - 支持多卡训练，请调整 `train.yaml` 中的 `use_multi_gpu` 参数和指定 GPU 配置。
   - 若使用上文下载的默认数据：推荐设置`train.yaml` 中的`learning_rate`为`2e-5`。设置`num_train_epochs`为`10`。
   - 提示：数据量较小时建议减小学习率，以避免参数更新过快；数据量较大时可以尝试增大学习率，加速模型收敛。可结合wandb监控训练过程loss（设置`config.yaml`中的`use_wandb`参数为True），调整学习率与训练轮数，以达到最佳效果。

2. **BiLSTM-CRF**  
   ```bash
   python run_lstmcrf.py
   ```  
   超参数设置详见 `lstmcrf.yaml`，其余参数在 `conf` 文件夹中调整。

3. **W2NER**  
   ```bash
   cd w2ner
   python run.py
   ```  
   参数配置详见 `model.yaml`，包括 GPU 指定参数 `device`。

#### 训练输出
- **日志**：保存在 `logs` 文件夹中。  
- **模型结果**：保存在 `checkpoints` 文件夹中。  
- **Batch Size**：建议 BERT 模型训练时，batch size 不小于 64。

#### 预测
使用以下命令进行预测：
```bash
python predict.py
```

---

### 样本自动化打标

如果只有文本数据和词典，没有规范的训练数据，可通过自动化打标方法生成弱监督的格式化训练数据。请确保：
1. 提供高质量的词典。  
2. 准备足够的文本数据。

相关详情请参考 [prepare-data](https://github.com/zjunlp/DeepKE/blob/main/example/ner/prepare-data/README.md)。
