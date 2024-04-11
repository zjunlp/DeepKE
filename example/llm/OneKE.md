<p align="center">
    <a href="https://github.com/zjunlp/deepke"> <img src="assets/oneke_logo.png" width="400"/></a>
<p>
<p align="center">  
    <a href="https://oneke.openkg.cn/">
        <img alt="Documentation" src="https://img.shields.io/badge/demo-website-blue">
    </a>
    <a href="https://pypi.org/project/deepke/#files">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/deepke">
    </a>
    <a href="https://github.com/zjunlp/DeepKE/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/zjunlp/deepke">
    </a>
    <a href="http://zjunlp.github.io/DeepKE">
        <img alt="Documentation" src="https://img.shields.io/badge/doc-website-red">
    </a>
</p>



<h1 align="center">
    <p>OneKE: A Bilingual Large Language Model for <br>Knowledge Extraction</p>
</h1>

- [什么是OneKE?](#什么是oneke)
- [OneKE是怎么训的?](#oneke是怎么训的)
- [快速上手OneKE](#快速上手oneke)
  - [环境安装](#环境安装)
  - [模型下载](#模型下载)
  - [快速运行](#快速运行)
- [专业使用OneKE](#专业使用oneke)
  - [OneKE指令格式](#oneke指令格式)
  - [OneKE指令格式转换](#oneke指令格式转换)
  - [定制化schema解释指令](#定制化schema解释指令)
  - [4bit量化OneKE](#4bit量化oneke)
- [继续训练](#继续训练)
- [项目贡献人员](#项目贡献人员)
- [引用](#引用)


## 什么是OneKE?

浙江大学与蚂蚁集团依托多年积累的知识图谱与自然语言处理技术，与2024年联合升级并发布新版中英双语知识抽取大模型OneKE。该模型采用基于Schema的轮询指令构造技术，专门针对提升大模型在结构化信息抽取的泛化能力进行了优化。

<p align="center" width="100%">
<a href="" target="_blank"><img src="assets/oneke.gif" alt="ChatGLM" style="width: 100%; min-width: 20px; display: block; margin: auto;"></a>
</p>

## OneKE是怎么训的?

OneKE主要聚焦基于Schema可泛化的信息抽取。由于现有的抽取指令数据存在格式不统一、数据噪音、多样性弱等问题，如下图所示OneKE采取了抽取指令的归一化与清洗、难负样本采样、基于Schema的轮询指令构造等技术，相关内容可查阅论文“**[IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus](https://arxiv.org/abs/2402.14710) [[Github](https://github.com/zjunlp/IEPile)]**”。


OneKE在零样本泛化性上与其他大模型的对比结果
* `NER-en`: CrossNER_AI、CrossNER_literature、CrossNER_music、CrossNER_politics、CrossNER_science
* `NER-zh`: WEIBONER、boson
* `RE-zh`: COAE2016、IPRE、SKE2020
* `RE-en`: FewRel、Wiki-ZSL
* `EE-en`: CrudeOilNews、WikiEvents、RAMS
* `EE-zh`: FewFC、CCF Law


<p align="center" width="50%">
<a href="" target="_blank"><img src="assets/oneke_results.png" alt="OneKE" style="width: 50%; min-width: 20px; display: block; margin: auto;"></a>
</p>


## 快速上手OneKE


### 环境安装

```bash
conda create -n deepke-llm python=3.9
conda activate deepke-llm
pip install -r requirements.txt
```

注意！！是example/llm文件夹下的 `requirements.txt`


### 模型下载

[HuggingFace](https://huggingface.co/zjunlp/OneKE), [ModelScope](https://modelscope.cn/models/ZJUNLP/OneKE), [WiseModel](https://wisemodel.cn/models/zjunlp/OneKE)  

### 快速运行

```python
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

model_path = 'zjunlp/OneKE'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()


system_prompt = '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
sintruct = "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}"
sintruct = '[INST] ' + system_prompt + sintruct + '[/INST]'

input_ids = tokenizer.encode(sintruct, return_tensors="pt")
input_length = input_ids.size(1)
generation_output = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_length=1024, max_new_tokens=512, return_dict_in_generate=True))
generation_output = generation_output.sequences[0]
generation_output = generation_output[input_length:]
output = tokenizer.decode(generation_output, skip_special_tokens=True)

print(output)
```


## 专业使用OneKE

训练和推理建议至少具备**20GB的显存**


### OneKE指令格式

在OneKE中 **`instruction`** 的格式采用了类JSON字符串的结构，实质上是一种字典类型的字符串。它由以下三个字段构成：
(1) **`'instruction'`**，即任务描述，以自然语言指定模型扮演的角色以及需要完成的任务；
(2) **`'schema'`**，这是一份需提取的标签列表，明确指出了待抽取信息的关键字段，反应用户的需求，是动态可变的；
(3) **`'input'`**，指的是用于信息抽取的源文本。


以下是各个任务的指令示例:

<details>
  <summary><b>实体命名识别(NER)</b></summary>

```json
{
	"instruction": "你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。",
	"schema": ["人名", "学历", "职位", "国籍"],
	"input": "刘志坚先生：1956年出生，中国国籍，无境外居留权，中共党员，大专学历，高级经济师。"
}
```

</details>


<details>
  <summary><b>关系识别(RE)</b></summary>

```json
{
	"instruction": "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。",
	"schema": ["父亲", "丈夫", "邮政编码", "母亲"],
	"input": "于是丁龙拿出自己的毕生积蓄12000美元，在19世纪末的12000美元无疑是一笔巨款，再加上卡朋蒂埃的捐助，两人一起资助哥伦比亚大学的汉学研究"
}
```

</details>



<details>
  <summary><b>知识图谱构建(KGC)</b></summary>

```json
{
    "instruction": "你是一个图谱实体知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息，不存在的属性不输出, 属性存在多值就返回列表，并输出为可解析的json格式。", 
    "schema": [
    {
        "entity_type": "人物", 
        "attributes": [ "中文名","英文名","祖籍","出生日期","出生地点","职业", "毕业学校", "作品","奖项"]
    }
    ], 
    "input": "周杰伦（Jay Chou），1979年1月18日出生于台湾省新北市，祖籍福建省泉州市永春县，华语流行乐男歌手、音乐人、演员、导演、编剧，毕业于淡江中学。2000年，发行个人首张音乐专辑《Jay》。2001年，凭借专辑《范特西》奠定其融合中西方音乐的风格。2002年，举行“The One”世界巡回演唱会；同年，凭借歌曲《爱在西元前》获得第13届台湾金曲奖最佳作曲人奖。"
}
```

</details>



<details>
  <summary><b>事件抽取(EE)</b></summary>

```json
{
    "instruction": "你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件，不存在的事件返回空列表，不存在的论元返回NAN，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。",
    "schema": [
        {
            "event_type": "财经/交易-加息",
            "trigger": True,
            "arguments": [
                "时间"
            ]
        },
        {
            "event_type": "财经/交易-降息",
            "trigger": True,
            "arguments": [
                "降息幅度"
            ]
        },
        {
            "event_type": "财经/交易-涨价",
            "trigger": True,
            "arguments": [
                "涨价方"
            ]
        },
        {
            "event_type": "财经/交易-降价",
            "trigger": True,
            "arguments": [
                "降价物",
                "时间"
            ]
        }
    ],
    "input": "AI风控解决方案供应商维择科技获数千万美元C+轮融资"
}
```

</details>



<details>
  <summary><b>事件触发词识别(EET)</b></summary>

```json
{
  "instruction": "你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件类型及事件触发词，不存在的事件返回空列表。请按照JSON字符串的格式回答。", 
  "schema": ["组织关系-解散", "组织关系-裁员", "组织关系-解雇", "竞赛行为-晋级"], 
  "input": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！"
}
```

</details>


<details>
  <summary><b>事件论元抽取(EEA)</b></summary>

```json
{
  "instruction": "你是专门进行事件论元提取的专家。请从input中抽取出符合schema定义的事件论元及论元角色，不存在的论元返回NAN或空字典，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。", 
  "schema": [{"event_type": "组织关系-辞/离职", "arguments": ["离职者", "时间", "原所属组织"]}], 
  "input": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！"
}
```

</details>





### OneKE指令格式转换

**指令列表**: 
```python
instruction_mapper = {
    'NERzh': "你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。",
    'REzh': "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。",
    'EEzh': "你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件，不存在的事件返回空列表，不存在的论元返回NAN，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。",
    'EETzh': "你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件类型及事件触发词，不存在的事件返回空列表。请按照JSON字符串的格式回答。",
    'EEAzh': "你是专门进行事件论元提取的专家。请从input中抽取出符合schema定义的事件论元及论元角色，不存在的论元返回NAN或空字典，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。",
    'KGzh': '你是一个图谱实体知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息，不存在的属性不输出, 属性存在多值就返回列表，并输出为可解析的json格式。',
    'NERen': "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.",
    'REen': "You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.",
    'EEen': "You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.",
    'EETen': "You are an expert in event extraction. Please extract event types and event trigger words from the input that conform to the schema definition. Return an empty list for non-existent events. Please respond in the format of a JSON string.",
    'EEAen': "You are an expert in event argument extraction. Please extract event arguments and their roles from the input that conform to the schema definition, which already includes event trigger words. If an argument does not exist, return NAN or an empty dictionary. Please respond in the format of a JSON string.", 
    'KGen': 'You are an expert in structured knowledge systems for graph entities. Based on the schema description of the input entity type, you extract the corresponding entity instances and their attribute information from the text. Attributes that do not exist should not be output. If an attribute has multiple values, a list should be returned. The results should be output in a parsable JSON format.',
}
```

各个任务的推荐**切分长度**:

```python
split_num_mapper = {
    'NER':6, 'RE':4, 'EE':4, 'EET':4, 'EEA':4, 'KG':1
}
```

由于一次性预测标签集中的所有schema难度过大, 且不易于扩展, 因此OneKE在训练时采用了轮询方式, 对指令中的schema询问数量进行了切分, 每次询问固定数量的schema, 因此一条数据如果其标签集过长, 将会被切分成多条指令轮流询问模型。



**schema格式**:
```python
NER: ["人名", "学历", "职位", "国籍"]   # 字符串列表
RE: ["父亲", "丈夫", "邮政编码", "母亲"]   # 字符串列表
EE: [{"event_type": "财经/交易-加息", "trigger": True, "arguments": ["时间"]}, {"event_type": "财经/交易-降息", "trigger": True, "arguments": ["降息幅度"]}]  # 字典列表, "event_type"是字符串, "trigger"是bool, "arguments"是列表
EET: ["组织关系-解散", "组织关系-裁员", "组织关系-解雇", "竞赛行为-晋级"]    # 字符串列表
EEA: [{"event_type": "财经/交易-加息", "arguments": ["时间"]}, {"event_type": "财经/交易-降息", "arguments": ["降息幅度"]}]  # 字典列表, "event_type"是字符串, "arguments"是列表
```


下面是简易的**轮询指令生成**脚本:
```python
def get_instruction(language, task, schema, input):
    sintructs = []
    split_num = split_num_mapper[task]
    if type(schema) == dict:
        sintruct = json.dumps({'instruction':instruction_mapper[task+language], 'schema':schema, 'input':input}, ensure_ascii=False)
        sintructs.append(sintruct)
    else:
        split_schemas = [schema[i:i+split_num] for i in range(0, len(schema), split_num)]
        for split_schema in split_schemas:
            sintruct = json.dumps({'instruction':instruction_mapper[task+language], 'schema':split_schema, 'input':input}, ensure_ascii=False)
            sintructs.append(sintruct)
    return sintructs
```

更详细的数据转换可参考[InstructKGC/README_CN.md/2.3测试数据转换](./InstructKGC/README_CN.md/#23测试数据转换)


下面是使用上述简易脚本的示例:

```python
task = 'NER'
language = 'en'
schema = ['person', 'organization', 'else', 'location']
split_num = split_num_mapper[task]
split_schemas = [schema[i:i+split_num] for i in range(0, len(schema), split_num)]
input = '284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )'
sintructs = []
for split_schema in split_schemas:
    sintruct = json.dumps({'instruction':instruction_mapper[task+language], 'schema':split_schema, 'input':input}, ensure_ascii=False)
    sintructs.append(sintruct)
```

> '{"instruction": "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.", "schema": ["person", "organization", "else", "location"], "input": "284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )"}'


### 定制化schema解释指令

```json
{
  "instruction": "你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。", 
  "schema": {
    "职位": "实体类型描述个人或群体的职业或职务，包括特定角色名称如'制片方'，'报分员'，'苦行僧'，'油画家'。", 
    "景点": "景点实体类型包括建筑物、博物馆、纪念馆、美术馆、河流、山峰等。代表性实体有五角大楼、泰特当代美术馆、郑成功纪念馆、都喜天阙、巴里卡萨、罗博河、gunungbatur、愚公移山LIVE、徐悲鸿纪念馆、杜莎夫人蜡像馆等。", 
    "公司": "公司是一个实体类型，代表任何法人实体或商业组织。这个类型的实体可以是餐饮集团，制造商，零售商，酒店，银行，设计院等各种类型的公司。例如：'香格里拉酒店集团', 'JVC', '上海酷蕾专业电竞外设装备店', 'k2&bull;海棠湾', '武钢', 'louisvuitton', '苏格兰银行', '北京市建筑设计研究院', '7天', '万科集团'。", 
    "地址": "地址实体是指具有地理位置信息的实体，它可以代表一个国家、城市、区域、街道等具体的地方或者一个抽象的地理区域。例如：'曼哈顿下城区东南尖上的河边码头', '图阿普谢', '意大利威尼斯水乡', '湖州温泉高尔夫球场', '北卡罗来纳州', '京津区域', '开心网吧', '颐年护理院', '上塘镇浦东', '内蒙古自治区赤峰市'等。", 
    "组织机构": "组织机构实体是指集体性质的组织，比如公司、商铺、俱乐部、学校等。它们在社会和经济活动中扮演一定角色，并拥有一定的人格权。", 
    "电影": "电影实体包括中文或英文电影名称，有时也包括电影人物角色名。"
  }, 
  "input": "我很难想象在另外一个项目再做一个海渔广场，当时我们拿到这个项目的时候，我正好在三亚，"
}
```


<details>
  <summary><b>关系抽取(RE)解释指令</b></summary>

```json
{
    "instruction": "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。",
    "schema": {
        "民族": "民族",
        "毕业院校": "该关系类型描述的是人物和其毕业学院之间的关系，人物是主体，毕业学院是客体。通过识别出文本中的人物名称和学校名称，然后通过词语的组合、上下文情境等信息，分析出人物和学院之间的毕业关系。",
        "主演": "这是一个描述影视作品与其主要演员之间关系的类型，主体是影视作品，客体是演员。在一个有效的'主演'关系中，演员（客体）在影视作品（主体）中担任重要的角色。",
        "父亲": "这个关系类型用来表示人物关系中的父亲和子女之间的亲属关系，即父亲是子女的出生或抚养者。在关系三元组中，'父亲'这个关系类型的主体是子女，客体是父亲。"
    },
    "input": "古往今来，能饰演古龙小说人物“楚留香”的，无一不是娱乐圈公认的美男子，2011年，36岁的张智尧在《楚留香新传》里饰演楚留香，依旧帅得让人无法自拔"
}
```

</details>


<details>
  <summary><b>事件抽取(EE)解释指令</b></summary>

```json
{
    "instruction": "你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件，不存在的事件返回空列表，不存在的论元返回NAN，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。",
    "schema": {
        "财经/交易-上市": {
            "财经/交易-上市": "金融实体在证券市场上进行上市的行为主要涉及公司、股票等。 正样本包括公司或股票上市的具体信息，负样本则与此类活动无关。",
            "trigger": True,
            "arguments": {
                "融资金额": "指的是公司在上市事件中筹集的资金总额。它综合了所有股份发行的收益，并以货币计量，包括但不限于'亿'、'万'、'美元'、'人民币'等单位。",
                "时间": "描述上市事件发生的具体时间，可以是具体日期或相对时间，也可以包含地点信息和具体的日子和星期。",
                "上市企业": "指的是在某次上市事件中，进行首次公开招股或已经在交易市场挂牌的公司或企业。例如：'上海复宏汉霖生物技术股份'、'三只松鼠'、'宝信软件'、'小熊电器'、'晋商银行'、''人造肉第一股'Beyond Meat(BYND)'、'游戏直播平台斗鱼'、'快餐帝国'以及'自动驾驶激光雷达厂商Velodyne'等。",
                "地点": "财经或交易事件发生的具体地点，比如城市、建筑物或房间。"
            }
        },
        "组织关系-辞/离职": {
            "组织关系-辞/离职": "事件类型'组织关系-辞/离职'是指个人或组织成员与所属组织关系变动的情况，主要包括'辞职'、'请辞'、'卸任'、'离队'、'退休'、'离开'等。常发生在高层人事变动、政府官员变动或运动员转会等场景。例如：'李楠宣布辞职'、'于旭波上任董事会主席3个月就辞职 陈朗接棒'。",
            "trigger": True,
            "arguments": {
                "离职者": "指在组织关系-辞/离职事件中，主动或被动地离开原来的职务或工作岗位的个体或群体，可以是一个人，也可以是一组人，如：'财政大臣', '90后邵阳隆回小伙欧阳恩和', '熊晓鸽', '*ST长生两位副总经理', '杨涛', '飞行员马强', 'HE WEI', '百度5名高管', '优信集团首席运营官彭惟廉', '建科院证券事务代表舒彦铭'等。",
                "时间": "表示辞/离职事件发生的具体时间点或时间段，一般包括具体的日期、周数、时间等信息。如'9月19号', '6月29日晚', '本周六', '7月9日上午10:30', '6月12日早晨', '4月9日', '9月10号', '当地时间周日', '九月十二日', '10月15日上午十点'等。"
            }
        },
        "财经/交易-加息": {
            "财经/交易-加息": "该事件描述银行或金融机构提高利率以收紧货币供应。典型触发词是'加息'。 '加息'表明了财经/交易-加息事件的发生。",
            "trigger": True,
            "arguments": {
                "加息幅度": "加息幅度通常表现为一个百分比或者基点，表示在加息事件中，利率提高的程度或者范围。例如：'至5.75%'，'25基点'，'基准利率从0.25%上升至0.5%'，'25个基点'。",
                "加息机构": "加息机构是指在财经/交易-加息事件中，决定或者执行加息政策的具有货币政策调控权的金融机构，比如各国的中央银行（如英央行、美联储、欧洲央行等）或是金融机构（如英格兰银行）。",
                "时间": "表示财经/交易-加息事件发生的具体日期或时间段，如'6月18日上午'、'1月24日'、'三个月后'等，具体表达形式既包括精确到分钟的时间如'2018年12月28日11时'，也包括相对时间如'昨天（2日）'和特殊的时间表达如'中秋档'等。"
            }
        },
        "组织关系-解约": {
            "组织关系-解约": "合同被取消或终止的情况通常在商业、娱乐或体育领域中发生。触发词有'离开', '交易', '裁掉', '合约到期', '解除合约', '贱卖', '解除', '送出', '解约'等。正例包括'彭昱畅解除合约'和'蒋梦婕解约后近破产'，负例如'费德勒退出了比赛'。",
            "trigger": True,
            "arguments": {
                "被解约方": "在组织关系解约事件中的角色，是被解除协议或合同关系的一方，可能是个人或组织，如运动员、电影制作方、公司等。例如，'7届全明星得主乔-约翰逊'，'《小小的愿望》片方'，'猛龙', '三星'等。"
            }
        }
    },
    "input": "8月20日消息，据腾讯新闻《一线》报道，知情人士表示，为了控制成本支出，蔚来计划将美国分公司的人员规模除自动驾驶业务相关人员外，减少至200人左右。截至美国时间8月16日，蔚来位于美国硅谷的分公司已裁减100名员工。"
}
```

</details>



<details>
  <summary><b>知识图谱构建(KGC)解释指令</b></summary>

```json
{
  "instruction": "你是一个图谱实体知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息，不存在的属性不输出, 属性存在多值就返回列表，并输出为可解析的json格式。", 
  "schema": [
    {
      "entity_type": "人物",
      "attributes": {
        "中文名": "人物中文名字", 
        "英文名": "人物的英文名", 
        "祖籍": "人物的祖籍地址",
        "出生日期": "生日、出生年月日", 
        "出生地点": "出生的地点、行政区",
        "职业": "人物的职业、职务、身份",
        "毕业学校": "就读毕业的中学、大学、高校",
        "作品": "专辑、歌曲、小说、出版书籍、参演影视作品等",
        "奖项": "人物所获得的各种奖项和荣誉称号"}
    }
  ], 
  "input": "周杰伦（Jay Chou），1979年1月18日出生于台湾省新北市，祖籍福建省泉州市永春县，华语流行乐男歌手、音乐人、演员、导演、编剧，毕业于淡江中学。2000年，发行个人首张音乐专辑《Jay》。2001年，凭借专辑《范特西》奠定其融合中西方音乐的风格。2002年，举行“The One”世界巡回演唱会；同年，凭借歌曲《爱在西元前》获得第13届台湾金曲奖最佳作曲人奖。"
}
```

</details>



### 4bit量化OneKE

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config=BitsAndBytesConfig(     
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map="auto", 
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

从输出文本中提取结构并评估可参考[InstructKGC/README_CN.md/7.评估](./InstructKGC/README_CN.md/#🧾-7评估)



## 继续训练

继续训练OneKE可参考[InstructKGC/4.9领域内数据继续训练](./InstructKGC/README_CN.md/#49领域内数据继续训练)



## 项目贡献人员

张宁豫、桂鸿浩、袁琳、孙梦姝、徐军、王昊奋、梁磊、陈华钧


## 引用

如果您使用了OneKE， 烦请引用下列论文: 

```bibtex
@article{DBLP:journals/corr/abs-2402-14710,
  author       = {Honghao Gui and
                  Lin Yuan and
                  Hongbin Ye and
                  Ningyu Zhang and
                  Mengshu Sun and
                  Lei Liang and
                  Huajun Chen},
  title        = {IEPile: Unearthing Large-Scale Schema-Based Information Extraction
                  Corpus},
  journal      = {CoRR},
  volume       = {abs/2402.14710},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2402.14710},
  doi          = {10.48550/ARXIV.2402.14710},
  eprinttype    = {arXiv},
  eprint       = {2402.14710},
  timestamp    = {Tue, 09 Apr 2024 07:32:43 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2402-14710.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

