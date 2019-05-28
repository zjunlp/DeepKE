# coding=utf-8

import re
import json
import torch
import jieba
from tqdm import tqdm
from vocab import Vocab
from utils import set_seed, pad
from config import config

set_seed(config.seed)


class CustomDataset(object):
    """定制 Dataset

    Args:
        fp(str): 数据集地址
        min_count(int, optional): 在构建词典时，筛选词语的词频，当词频低于 min_count 时，词语不计入词典。
                                  默认值为 2。

    Output:
        train_x: 处理后的数据，用于输入到 dataloader 中
        train_l: train_x 对应的每条句子长度，用于在输入 rnn 前将真实长度还原
        train_y: train_x 对应的关系 label
    """

    def __init__(self, fp, min_count=2):
        """定制 Dataset

        Args:
            fp(str): 数据集地址
        """
        self.fp = fp
        self.min_count = min_count

        # word2id
        vocab = Vocab(self.min_count, data_path=self.fp)
        self.word2id = vocab.word2id

        # 构建 rel2id
        # 提取所有关系，构建 rel2id
        self.relations = set()
        with open(self.fp, encoding='utf-8') as f:
            for line in f:
                data = json.loads(str(line))
                self.relations.add(data['relation'])
        self.relations = list(self.relations)
        self.rel2id = {i: j for j, i in enumerate(self.relations)}

        # 需要使用的向量
        self.data_x_word = []  # 单词向量，2维，每一项都是数组
        self.data_x_pos1 = []  # 实体1位置向量，2维，每一项都是数组
        self.data_x_pos2 = []  # 实体2位置向量，2维，每一项都是数组
        self.data_x_len = []  # 句子长度向量，1维
        self.data_y = []  # 句子对应的关系向量，1维
        self.max_len = 0
        self.pos_max_len = 100  # 为保证 pos 都大于0，将所有pos值增加一定值

        with open(self.fp, encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(str(line))
                instance = data['text'].strip()
                punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
                re_punc = "[{}]+".format(punc)
                instance = re.sub(re_punc, "", instance).strip()
                instance = instance.split(data['entity1'], 1)
                instance = [instance[0], 'E1', instance[1]]
                if instance[0].find(data['entity2']) > -1:
                    instance[0] = instance[0].split(data['entity2'], 1)
                    ins = list(jieba.cut(instance[0][0]))
                    ins.append('E2')
                    ins.extend(list(jieba.cut(instance[0][1])))
                    ins.append('E1')
                    ins.extend(list(jieba.cut(instance[2])))
                else:
                    instance[2] = instance[2].split(data['entity2'], 1)
                    ins = list(jieba.cut(instance[0]))
                    ins.append('E1')
                    ins.extend(list(jieba.cut(instance[2][0])))
                    ins.append('E2')
                    ins.extend(list(jieba.cut(instance[2][1])))
                ins_len = len(ins)
                e1idx = ins.index('E1')
                e2idx = ins.index('E2')
                ins_pos12id = [
                    i - e1idx + self.pos_max_len for i in range(ins_len)
                ]
                ins_pos22id = [
                    i - e2idx + self.pos_max_len for i in range(ins_len)
                ]
                self.data_x_pos1.append(ins_pos12id)
                self.data_x_pos2.append(ins_pos22id)
                self.data_x_len.append(ins_len)
                x_vec = [self.word2id.get(c, 0) for c in ins]
                self.data_x_word.append(x_vec)
                data_y_ins = data['relation']
                y_vec = self.rel2id[data_y_ins]
                self.data_y.append(y_vec)
                if ins_len > self.max_len:
                    self.max_len = ins_len
        # 需要词向量和位置向量
        self.data_x = [
            self.data_x_word[i] + self.data_x_pos1[i] + self.data_x_pos2[i]
            for i in range(len(self.data_x_len))
        ]
        self.train_x = []
        self.train_y = []
        self.train_l = []
        for i in range(len(self.data_x_len)):
            self.train_x.append(pad(self.data_x[i], self.max_len * 3))
            self.train_y.append(self.data_y[i])
            self.train_l.append(len(self.data_x[i]))

        # 将所有变量变成 torch tensor
        self.train_x = torch.tensor(self.train_x, dtype=torch.int64)
        self.train_y = torch.tensor(self.train_y, dtype=torch.int64)
        self.train_l = torch.tensor(self.train_l, dtype=torch.int64)


if __name__ == '__main__':
    # 单元测试
    dataset = CustomDataset(fp='data/origin/train.txt', min_count=2)
    print(dataset.train_x.size())
