## 快速上手

<p align="left">
    <b> <a href="./README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

```bash
cd ..
pip install -r requirements
```

## 克隆代码

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ee/standard/degree
```

## 数据处理

- 通过更改`.conf/config.yaml`中的参数来选择数据划分的比例，默认的设定为`001`。
- 根据[这里](./data/ACE/README.md)的文档来得到`ACE`数据集。
- 运行`python generate_data.py`命令来得到对应划分处理后的数据，默认设定下处理好的数据将会被放在`./processed_data/ace_001`文件夹下。

## 训练
- 参数、模型路径以及一些参数都在`./conf/config.yaml`文件夹下，在训练之前可以对他们进行更改。
- 然后运行下述命令
```bash
python run.py
```

## 预测
- 将 `./conf/config.yaml`中的`e2e_model` 参数修改成你自己的训练得到的模型的路径，然后运行下面的命令。
```bash
python predict.py
```