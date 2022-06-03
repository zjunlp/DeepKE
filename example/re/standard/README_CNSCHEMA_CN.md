
# cnSchema

我们预先处理数据集，并将其关系类型与实体类型与[cnSchema](https://github.com/OpenKG-ORG/cnSchema)对齐。cnSchema是面向中文信息处理，利用先进的知识图谱、自然语言处理和机器学习技术，融合结构化与文本数据，支持快速领域知识建模，支持跨数据源、跨领域、跨语言的开放数据自动化处理，为智能机器人、语义搜索、智能计算等新兴应用市场提供schema层面的支持与服务。

cnSchema基于的原则
* 完全开放，源自schema.org，OpenKG自主发布的Web Schema
* 立足中文，对接世界
* 面向应用，支持开放数据生态
* 社区共识，知识图谱专家指导

# 关系类型
经过对齐后数据集中所拥有的关系类型如下

| 序号  | 头实体类型 | 尾实体类型| 关系| 序号  | 头实体类型 | 尾实体类型| 关系| 
| --- | :--- | :---: | --- | --- | :--- | :---: | --- | 
| 1  | 地点| 人物 | 祖籍 |2  | 人物| 人物 | 父亲 |
| 3  | 地点| 企业 | 总部地点 |4  | 地点| 人物 | 出生地 |
| 5  | 目| 生物 | 目 |6  | Number| 行政区 | 面积 |
| 7  | Text| 机构 | 简称 |8  | Date| 影视作品 | 上映时间 |
| 9  | 人物| 人物 | 妻子 |10  | 音乐专辑| 歌曲 | 所属专辑 |
| 11  | Number| 企业 | 注册资本 |12  | 城市| 国家 | 首都 |
| 13  | 人物| 影视作品 | 导演 |14  | Text| 历史人物 | 字 |
| 15  | Number| 人物 | 身高 |16  | 企业| 影视作品 | 出品公司 |
| 17  | Number| 学科专业 | 修业年限 |18  | Date| 人物 | 出生日期 |
| 19  | 人物| 影视作品 | 制片人 |20  | 人物| 人物 | 母亲 |
| 21  | 人物| 影视作品 | 编辑 |22  | 国家| 人物 | 国籍 |
| 23  | 人物| 影视作品 | 编剧 |24  | 网站| 网站小说 | 连载网络 |
| 25  | 人物| 人物 | 丈夫 |26  | Text| 历史人物 | 朝代 |
| 27  | Text| 人物 | 民族 |28  | Text| 历史人物 | 朝代 |
| 29  | 出版社| 书籍 | 出版社 |30  | 人物| 电视综艺 | 主持人 |
| 31  | Text| 学科专业 | 专业代码 |32  | 人物| 歌曲 | 歌手 |
| 33  | 人物| 歌曲 | 作曲 |34  | 人物| 网络小说 | 主角 |
| 35  | 人物| 企业 | 董事长 |36  | Date| 机构 | 成立时间 |
| 37  | 学校| 人物 | 毕业院校 |38  | Number| 机构 | 占地面积 |
| 39  | 语言| 国家 | 官方语言 |40  | Text| 行政区 | 人口数量 |
| 41  | Number| 行政区 | 人口数量 |42  | 城市| 景点 | 所在城市 |
| 43  | 人物| 图书作品 | 作者 |44  | Date| 企业 | 成立时间 |
| 45  | 人物| 歌曲 | 作曲 |46  | 人物| 行政区 | 气候 |
| 47  | 人物| 电视综艺 | 嘉宾 |48  | 人物| 影视作品 | 主演 |
| 49  | 作品| 影视作品 | 改编自 |50  | 人物| 企业 | 创始人 |





在这之上使用[`chinese-bert-wwm`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)和[`chinese-roberta-wwm-ext`](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)为基础训练了DeepKE-cnschema(RE)模型。模型所使用的超参数为所给的参数。最终经过训练后可以得到如下表的效果

<table>
	<tr>
		<th>模型</th>
		<th>P</th>
		<th>R</th>
		<th>F1</th>
	</tr>
  <tr>
		<td>chinese-roberta-wwm-ext(macro)</td>
		<td>0.8761</td>
		<td>0.8598</td>
		<td>0.8665</td>
	</tr>
  <tr>
		<td>chinese-bert-wwm(macro)</td>
		<td>0.8742</td>
		<td>0.8582</td>
		<td>0.8639</td>
	</tr>
	
</table>

# 预测
使用者可以直接下载[模型](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)使用,步骤如下：

1、修改 `predict.yaml`中的参数`fp`为下载文件的路径，`embedding.yaml`中`num_relations`为51

2、进行预测。需要预测的文本及实体对通过终端返回给程序。

```bash
python predict.py
```
   
# 训练

如果需要使用其他模型进行训练，步骤如下：

1、也可先下载[数据集](https://drive.google.com/drive/folders/1wb_QIZduKDwrHeri0s5byibsSQrrJTEv)，将其重命名为`data`

2、将`conf`文件夹中的`train.yaml`为`lm`,`lm.yaml`中的`lm_file`修改为指定预训练模型，`embedding.yaml`中`num_relations`为51

3、进行训练。
```bash
python run.py
```

# 样例
使用训练好的模型，运行```python predict.py```后，只需输入的句子为“歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版”，给定的实体对为“男人的爱”和“人生长路”，可得到结果，最终抽取出的关系为经过cnschema对齐后的“所属专辑”。


## 代码部分

只需将predict.py的_get_predict_instance函数修改成如下或不使用范例，即可修改文本进行预测
```python
def _get_predict_instance(cfg):
    flag = input('是否使用范例[y/n]，退出请输入: exit .... ')
    flag = flag.strip().lower()
    if flag == 'y' or flag == 'yes':
        sentence = '歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版'
        head = '男人的爱'
        tail = '人生长路'
        head_type = ''
        tail_type = ''
    elif flag == 'n' or flag == 'no':
        sentence = input('请输入句子：')
        head = input('请输入句中需要预测关系的头实体：')
        head_type = input('请输入头实体类型（可以为空，按enter跳过）：')
        tail = input('请输入句中需要预测关系的尾实体：')
        tail_type = input('请输入尾实体类型（可以为空，按enter跳过）：')
    elif flag == 'exit':
        sys.exit(0)
    else:
        print('please input yes or no, or exit!')
        _get_predict_instance()

    instance = dict()
    instance['sentence'] = sentence.strip()
    instance['head'] = head.strip()
    instance['tail'] = tail.strip()
    if head_type.strip() == '' or tail_type.strip() == '':
        cfg.replace_entity_with_type = False
        instance['head_type'] = 'None'
        instance['tail_type'] = 'None'
    else:
        instance['head_type'] = head_type.strip()
        instance['tail_type'] = tail_type.strip()

    return instance
```

最终输出结果

```bash
“男人的爱”和“人生长路”在句中关系为“所属专辑”，置信度为0.99
```
## 演示gif
具体流程如下gif所示：

<img src="demo.gif" />
