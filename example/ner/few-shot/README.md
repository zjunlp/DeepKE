## 快速上手

### 环境依赖

> python == 3.8

- torch == 1.7
- tensorboardX ==2.4
- transformers == 3.4.0
- deepke 

### 克隆代码
```
git clone git@github.com:zjunlp/DeepKE.git
```
### 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

### 使用数据进行训练预测

- 存放数据：在 `data` 文件夹下存放训练数据。包含conll2003，mit-movie，mit-restaurant和atis等数据集。

- conll2003包含以下数据：

  - `train.txt`：存放训练数据集

  - `dev.txt`：存放验证数据集

  - `test.txt`：存放测试数据集

  - `indomain-train.txt`：存放indomain数据集

- mit-movie, mit-restaurant和atis包含以下数据：

  - `k-shot-train.txt`：k=[10, 20, 50, 100, 200, 500]，存放训练数据集

  - `test.txt`：存放测试数据集


- 开始训练：模型加载和保存位置以及配置可以在shell脚本中修改
  
  - 训练conll2003：` python run.py ` (训练所用到参数都在conf文件夹中，修改即可)

  - 进行few-shot训练：` python run.py +train=few_shot ` (若要加载模型，修改few_shot.yaml中的load_path)


- 每次训练的日志保存在 `logs` 文件夹内，模型结果保存目录可以自定义。

- 进行预测：在config.yaml中加入 - predict ， 再在predict.yaml中修改load_path为模型路径以及write_path为预测结果保存路径，再` python predict.py `



