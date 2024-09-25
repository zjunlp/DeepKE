<p align="left">
	<b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/prepare-data/README_CN.md">简体中文</a></b>
</p>

We provide a simple **distant supervised** based tool to label relation labels for our RE tasks.

## Source File

We specify the source file (dataset to be labeled) as `.json` format and include **one** pair of entities, head entity and tail entity respectively. Each piece of data should contain at least the following five items: `sentence`, `head`, `tail`, `head_offset`, `tail_offset`. The detailed json pattern is as follows:

```json
[
  {
    "sentence": "This summer, the United States Embassy in Beirut, Lebanon, once again made its presence felt on the cultural scene by sponsoring a photo exhibition, an experimental jazz performance, a classical music concert and a visit from the Whiffenpoofs, Yale University's a cappella singers.",
    "head": "Lebanon",
    "tail": "Beirut",
    "head_offset": "50",
    "tail_offset": "42",
    //...
  },
  //... 
]
```

## Triple File

Entity pairs in source file will be matched with the triples in the triple file. The entity pairs will be labeled with the relation type if matched with the triples in triple file. If there is no triples match, the pairs will be labeled as `None` type.

We provide an [English](https://drive.google.com/drive/folders/1HHkm3RBI3okiu8jGs-Wn0vP_-0hz7pb2?usp=sharing) and a [Chinese](https://drive.google.com/file/d/1YpaMpivodG39p53MM9sMpB41q4EoiQhH/view?usp=sharing) triple file respectively. The English triple file comes from `NYT` dataset which contains the following relation types:

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

The Chinese triple file are from [here](https://github.com/DannyLee1991/ExtractTriples) with the following relation types:

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

You can also use your customized triple file, but the file format should be `.csv` and with the following parttern:

|  head   |  tail  |             rel             |
| :-----: | :----: | :-------------------------: |
| Lebanon | Beirut | /location/location/contains |
|   ...   |  ...   |             ...             |

## Ouput File

The output file names are `labeled_train.json`, `labeled_dev.json`, `labeled_test.json` for the `train`, `dev`, `test` dataset. The format of the output file is as follows:

```json
[
	{
    "sentence": "This summer, the United States Embassy in Beirut, Lebanon, once again made its presence felt on the cultural scene by sponsoring a photo exhibition, an experimental jazz performance, a classical music concert and a visit from the Whiffenpoofs, Yale University's a cappella singers.",
    "head": "Lebanon",
    "tail": "Beirut",
    "head_offset": "50",
    "tail_offset": "42",
    "relation": "/location/location/contains",
    //...
	},
  //...
]
```

We automatically split the source data into three splits with the rate 0.8:0.1:0.1. You can set your own split rate.

## Args Description

- `language`: `en` or `cn`
- `source_file`: data file to be labeled
- `triple_file`: triple file path
- `test_rate, dev_rate, test_rate`: The ratio of training_set, validation_set, and test_set (please make sure the sum is `1`, default `0.8:0.1:0.1`)

## Run

```bash
python ds_label_data.py --language en --source_file source_data.json --triple_file triple_file.csv
```

