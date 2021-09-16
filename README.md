<p align="center">
   <br>DeepKE</br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="https://github.com/zjunlp/DeepKE/blob/test_new_deepke/README_ENGLISH.md">English</a> 
    <p>
</h4>

<h3 align="center">
    <p>基于深度学习的开源中文知识图谱抽取框架</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/course_banner.png"></a>
</h3>

DeepKE 提供了多种知识抽取模型。

## 在线演示
演示的demo地址

1. RE 

   ```
   1.STANDARD
   
   2.FEW-SHOT
   
   3.DOCUMENT
   ```

2. NER

   ```
   STANDARD
   ```

3. AE
   ```
   STANDARD
   ```


## 快速上手

1. RE

   数据为csv文件，样式范例为：

   |                        Sentence                        | Relation |    Head    | Head_offset |    Tail    | Tail_offset |
   | :----------------------------------------------------: | :------: | :--------: | :---------: | :--------: | :---------: |
   | 《岳父也是爹》是王军执导的电视剧，由马恩然、范明主演。 |   导演   | 岳父也是爹 |      1      |    王军    |      8      |
   |  《九玄珠》是在纵横中文网连载的一部小说，作者是龙马。  | 连载网站 |   九玄珠   |      1      | 纵横中文网 |      7      |
   |     提起杭州的美景，西湖总是第一个映入脑海的词语。     | 所在城市 |    西湖    |      8      |    杭州    |      2      |
   
   具体流程请进入详细的README中，RE包括了以下三个子功能
   
   **[STANDARD](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/re/standard)**
   
   FEW-SHORT
   
   DOCUMENT

2. NER

   数据为txt文件

   中文样式范例为：

   |                           Sentence                           |           Person           |   Location   |          Organization          |
   | :----------------------------------------------------------: | :------------------------: | :----------: | :----------------------------: |
   | 本报北京9月4日讯记者杨涌报道：部分省区人民日报宣传发行工作座谈会9月3日在4日在京举行。 |            杨涌            |   北京，京   |            人民日报            |
   | 《红楼梦》是中央电视台和中国电视剧制作中心根据中国古典文学名著《红楼梦》摄制于1987年的一部古装连续剧，由王扶林导演，周汝昌、王蒙、周岭等多位红学家参与制作。 | 王扶林，周汝昌，王蒙，周岭 |      /       | 中央电视台，中国电视剧制作中心 |
   | 秦始皇兵马俑位于陕西省西安市，1961年被国务院公布为第一批全国重点文物保护单位，是世界八大奇迹之一。 |           秦始皇           | 陕西省西安市 |             国务院             |

   英文样式范例为：

   |                           Sentence                           |                Person                |     Location      |        Organization         |     Miscellaneous     |
   | :----------------------------------------------------------: | :----------------------------------: | :---------------: | :-------------------------: | :-------------------: |
   | Australian Tom Moody took six for 82 but Chris Adams, 123, and Tim O'Gorman, 109, took Derbyshire to 471 and a first innings lead of 233. | Tom Moody, Chris Adams, Tim O'Gorman |         /         |          Derbysire          |      Australian       |
   | Irene, a master student in Zhejiang University, Hangzhou, is traveling in Warsaw for Chopin Music Festival. |                Irene                 | Hangzhou, Warsaw  |     Zhejiang University     | Chopin Music Festival |
   | It was one o'clock when we left Lauriston Gardens and Sherlock Holmes led me to Metropolitan Police Service. |           Sherlock Holmes            | Lauriston Gardens | Metropolitan Police Service |           /           |

   具体流程请进入详细的README中：

   **[STANDARD](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/ner/standard)**

3. AE

    数据为csv文件，样式范例为：

   |                        Sentence                        | Attribute | Entity | Entity_offset | Attribute_value | Attribute_value_offset |
   | :----------------------------------------------------: | :-----: | :-----: | :---: | :-----: | :------: |
   | 张冬梅，女，汉族，1968年2月生，河南淇县人，1988年7月加入中国共产党，1989年9月参加工作，中央党校经济管理专业毕业，中央党校研究生学历 | 民族 | 张冬梅 | 0 | 汉族 | 6 |
   | 杨缨，字绵公，号钓溪，松溪县人，祖籍将乐，是北宋理学家杨时的七世孙  | 朝代 | 杨缨 | 0 | 北宋 | 22 |
   | 2014年10月1日许鞍华执导的电影《黄金时代》上映，冯绍峰饰演与之差别极大的民国东北爷们萧军，演技受到肯定| 上映时间 | 黄金时代 | 19 | 2014年10月1日 | 0 |
   
   具体流程请进入详细的README中:
   
   **[STANDARD](https://github.com/zjunlp/deepke/blob/test_new_deepke/example/ae/standard)**



## 模型架构
Deepke的架构图如下所示

<h3 align="center">
    <img src="pics/deepke.png">
</h3>


## 备注（常见问题）
1. 使用 Anaconda 时，建议添加国内镜像，下载速度更快。如[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

1. 使用 pip 时，建议使用国内镜像，下载速度更快，如阿里云镜像。

1. 安装后提示 `ModuleNotFoundError: No module named 'past'`，输入命令 `pip install future` 即可解决。

1. 使用语言预训练模型时，在线安装下载模型比较慢，更建议提前下载好，存放到 pretrained 文件夹内。具体存放文件要求见文件夹内的 readme.md。

## 致谢


## 引用
