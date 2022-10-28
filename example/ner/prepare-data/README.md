## Dict
Two entity Dicts (one in Chinese and one in English) are provided in advance, and the samples are automatically tagged using the entity dictionary + jieba part-of-speech tagging.

- In Chinese example dict, we adapt [People's Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily) dataset. It is a dataset for NER, concentrating on their types of named entities related to persons(PER), locations(LOC), and organizations(ORG).

- In English example dict，we adapt Conll dataset. It is a dataset for NER, concentrating on their types of named entities related to persons(PER), locations(LOC), organizations(ORG) and others(MISC).You can get the Conll dataset with the following command.

```shell
wget 120.27.214.45/Data/ner/few_shot/data.tar.gz
```

Pre-provided dict from Google Drive： [CN(vocab_dict_cn), EN(vocab_dict_en)](https://drive.google.com/drive/folders/1PGANizeTsvEQFYTL8O1jrDLZwk_MPqO0?usp=sharing)
From BaiduNetDisk ： [CN(vocab_dict_cn), EN(vocab_dict_en)](https://pan.baidu.com/s/1a07W42ZByeZ00MZp5pZgxg) (x7ba)

**If you need to build a domain self-built dictionary, please refer to the pre-provided dictionary format (csv)**

| Entity | Label |
|  ----  | ----  |
| Washington | LOC  |




## Environment
Implementation Environment:  
- jieba = 0.42.1

## Args Description

- `language`: `cn` or `en`
- `source_dir`: Corpus path (traverse all files in txt format under this folder, automatically mark line by line, the default is `source_data`)
- `dict_dir`: Entity dict path (defaults to `vocab_dict.csv`)
- `test_rate, dev_rate, test_rate`: The ratio of training_set, validation_set, and test_set (please make sure the sum is `1`, default `0.8:0.1:0.1`)

## run

- **Chinese**
```bash
python prepare_weaksupervised_data.py --language cn --dict_dir vocab_dict_cn.csv
```

- **English**
```bash
python prepare_weaksupervised_data.py --language en --dict_dir vocab_dict_en.csv
```
