<p align="left">
  <b> <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data/README.md">English</a> | 简体中文</b>
</p>

为了用户更好的使用`DeepKE`完成关系抽取任务，我们提供一个简单易用的基于远程监督的关系标注工具。

## 源文件

用户提供的源文件需要为`.json`形式，并且每条数据只包含一个实体对，分别为头实体和尾实体。数据中必须至少包含以下四个字段：sentence(句子), head(头实体), tail(尾实体), head_offset(头实体在句子中的偏移位置), tail_offset(尾实体在句子中的偏移位置)。具体的json字典形式如下：

```json
[
  {
    "sentence": "如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈",
    "head": "周星驰",
    "tail": "喜剧之王",
    "head_offset": "26",
    "tail_offset": "21",
    //...
	},
  //...
]
```

## 三元组文件

源文件中的实体对将会与三元组文件中的三元组进行字符串全匹配。如果匹配成功实体对将会被标记为三元组的关系标签，如果没有匹配项则被标记为`None`。

我们分解提供了一个[英文](https://drive.google.com/drive/folders/1HHkm3RBI3okiu8jGs-Wn0vP_-0hz7pb2?usp=sharing)和一个[中文](https://drive.google.com/file/d/1YpaMpivodG39p53MM9sMpB41q4EoiQhH/view?usp=sharing)三元组文件。英文三元组来自`NYT`数据集，包含如下关系类型：

```python
"/business/company/place_founded",
"/people/person/place_lived",
"/location/country/administrative_divisions",
"/business/company/major_shareholders",
"/sports/sports_team_location/teams",
"/people/person/religion",
"/people/person/place_of_birth",
"/people/person/nationality",
"/location/country/capital",
"/business/company/advisors",
"/people/deceased_person/place_of_death",
"/business/company/founders",
"/location/location/contains",
"/people/person/ethnicity",
"/business/company_shareholder/major_shareholder_of",
"/people/ethnicity/geographic_distribution",
"/people/person/profession",
"/business/person/company",
"/people/person/children",
"/location/administrative_division/country",
"/people/ethnicity/people",
"/sports/sports_team/location",
"/location/neighborhood/neighborhood_of",
"/business/company/industry"
```

中文三元组来自GitHub[链接](https://github.com/DannyLee1991/ExtractTriples)，包含如下关系类型：

```json
{"object_type": "地点", "predicate": "祖籍", "subject_type": "人物"}
{"object_type": "人物", "predicate": "父亲", "subject_type": "人物"}
{"object_type": "地点", "predicate": "总部地点", "subject_type": "企业"}
{"object_type": "地点", "predicate": "出生地", "subject_type": "人物"}
{"object_type": "目", "predicate": "目", "subject_type": "生物"}
{"object_type": "Number", "predicate": "面积", "subject_type": "行政区"}
{"object_type": "Text", "predicate": "简称", "subject_type": "机构"}
{"object_type": "Date", "predicate": "上映时间", "subject_type": "影视作品"}
{"object_type": "人物", "predicate": "妻子", "subject_type": "人物"}
{"object_type": "音乐专辑", "predicate": "所属专辑", "subject_type": "歌曲"}
{"object_type": "Number", "predicate": "注册资本", "subject_type": "企业"}
{"object_type": "城市", "predicate": "首都", "subject_type": "国家"}
{"object_type": "人物", "predicate": "导演", "subject_type": "影视作品"}
{"object_type": "Text", "predicate": "字", "subject_type": "历史人物"}
{"object_type": "Number", "predicate": "身高", "subject_type": "人物"}
{"object_type": "企业", "predicate": "出品公司", "subject_type": "影视作品"}
{"object_type": "Number", "predicate": "修业年限", "subject_type": "学科专业"}
{"object_type": "Date", "predicate": "出生日期", "subject_type": "人物"}
{"object_type": "人物", "predicate": "制片人", "subject_type": "影视作品"}
{"object_type": "人物", "predicate": "母亲", "subject_type": "人物"}
{"object_type": "人物", "predicate": "编剧", "subject_type": "影视作品"}
{"object_type": "国家", "predicate": "国籍", "subject_type": "人物"}
{"object_type": "Number", "predicate": "海拔", "subject_type": "地点"}
{"object_type": "网站", "predicate": "连载网站", "subject_type": "网络小说"}
{"object_type": "人物", "predicate": "丈夫", "subject_type": "人物"}
{"object_type": "Text", "predicate": "朝代", "subject_type": "历史人物"}
{"object_type": "Text", "predicate": "民族", "subject_type": "人物"}
{"object_type": "Text", "predicate": "号", "subject_type": "历史人物"}
{"object_type": "出版社", "predicate": "出版社", "subject_type": "书籍"}
{"object_type": "人物", "predicate": "主持人", "subject_type": "电视综艺"}
{"object_type": "Text", "predicate": "专业代码", "subject_type": "学科专业"}
{"object_type": "人物", "predicate": "歌手", "subject_type": "歌曲"}
{"object_type": "人物", "predicate": "作词", "subject_type": "歌曲"}
{"object_type": "人物", "predicate": "主角", "subject_type": "网络小说"}
{"object_type": "人物", "predicate": "董事长", "subject_type": "企业"}
{"object_type": "Date", "predicate": "成立日期", "subject_type": "机构"}
{"object_type": "学校", "predicate": "毕业院校", "subject_type": "人物"}
{"object_type": "Number", "predicate": "占地面积", "subject_type": "机构"}
{"object_type": "语言", "predicate": "官方语言", "subject_type": "国家"}
{"object_type": "Text", "predicate": "邮政编码", "subject_type": "行政区"}
{"object_type": "Number", "predicate": "人口数量", "subject_type": "行政区"}
{"object_type": "城市", "predicate": "所在城市", "subject_type": "景点"}
{"object_type": "人物", "predicate": "作者", "subject_type": "图书作品"}
{"object_type": "Date", "predicate": "成立日期", "subject_type": "企业"}
{"object_type": "人物", "predicate": "作曲", "subject_type": "歌曲"}
{"object_type": "气候", "predicate": "气候", "subject_type": "行政区"}
{"object_type": "人物", "predicate": "嘉宾", "subject_type": "电视综艺"}
{"object_type": "人物", "predicate": "主演", "subject_type": "影视作品"}
{"object_type": "作品", "predicate": "改编自", "subject_type": "影视作品"}
{"object_type": "人物", "predicate": "创始人", "subject_type": "企业"}
```

用户可以自定义三元组文件，但是需要设定文件形式为`.csv`并具有如下格式：

| 头实体 | 尾实体   | 关系 |
| ------ | -------- | ---- |
| 周星驰 | 喜剧之王 | 主演 |
| ...    | ...      | ...  |

## 输出文件

输出文件包含三个：`labeled_train.json`, `labeled_dev.json`, `labeled_test.json`，分别对应训练集、验证集和测试集。我们自动将源文件数据划分为三份，比例为0.8:0.1:0.1。输出文件的格式如下：

```json
[
	{
    "sentence": "如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈",
    "head": "周星驰",
    "tail": "喜剧之王",
    "head_offset": "26",
    "tail_offset": "21",
    "relation": "主演",
    //...
	},
  //...
]
```

## 参数描述

- `language`: 'en'对应英文数据，'cn'对应中文数据
- `source_file`: 需要标记的源文件
- `triple_file`: 三元组文件
- `test_rate, dev_rate, test_rate`:  训练集、验证集和测试集的划分比例，需要保证比例之和为1，默认为0.8:0.1:0.1。

## 运行

```bash
python ds_label_data.py --language en --source_file source_data.json --triple_file triple_file.csv
```

