## 快速上手

<p align="left">
    <b> <a href="./README.md">English</a> | 简体中文 </b>
</p>

### 环境依赖

- 首先创建Conda虚拟环境

- 安装环境依赖
  ```bash
  python==3.8
  pip install -r requirements.txt
  ```

## 克隆代码

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ee/standard
```

## 数据集

- `ACE`
根据[这里](./data/ACE/README.md)的文档进行处理。

- `DuEE`
根据[这里](./data/DuEE/README.md)的文档进行处理。

## 训练

在`./conf/train.yaml`中修改模型参数

- Trigger 触发词
  将`task_name`设置为`trigger`。
  可以通过更改`data_name`参数来选择不同的数据集。
  然后运行下述命令
  ```bash
  python run.py
  ```

- 事件角色
  在这里我们用正确的trigger训练事件元素抽取模型
  将`task_name`设置为`role`。
  可以通过更改`data_name`参数来选择不同的数据集。
  然后运行下述命令
  ```bash
  python run.py
  ```

## 预测

触发词的预测在训练的过程中会完成，预测的结果在`output_dir`中。在这里我们使用预测得到的触发词来抽取事件抽取元素。
在`./conf/predict.yaml`中修改模型参数。
然后运行下述命令
```bash
  python predict.py
```