## 快速上手

<p align="left">
    <b> <a href="./README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

请下载以下依赖:

```
python==3.8
Java 8
```

## 克隆代码

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ee/few-shot
```

## 使用pip安装

首先创建python虚拟环境，再进入虚拟环境

- 安装依赖: ```pip install -r requirements.txt```

## 训练以及预测

- 数据集
  - `ace05e`
  1. 跟据 [DyGIE++](https://github.com/dwadden/dygiepp#ace05-event)的预处理方式来的到数据
  2. 将处理好的数据放到具体目录 `data/raw_data/ace05e_dygieppformat`

  - `ace05ep`
  1. 从 [LDC](https://catalog.ldc.upenn.edu/LDC2006T06)下载ACE数据集
  2. 将处理好的数据放到具体目录 `data/raw_data/ace_2005`

  - `预处理`
  ```Bash
  cd data/
  bash ./scripts/process_ace05e.sh
  bash ./scripts/process_ace05ep.sh
  ```
  可以通过修改`conf/generate_data.yaml`中的参数来选择低资源的划分方式，然后执行下面的命令
  ```Bash
  cd ..
  python generate_data.py
  ```
  最终处理好的数据会存储在类似 `proceesed_data/degree_e2e_ace05ep_001`的目录下。


- 训练

  - 参数、模型路径以及一些参数都在`conf/train.yaml`文件夹下，在训练之前可以对他们进行更改。
  ```bash
  python run.py
  ```

- 预测

  - 将 `conf/predict.yaml`中的`e2e_model` 参数修改成你自己的训练得到的模型的路径，然后运行下面的命令。
  ```bash
  python predict.py
  ```