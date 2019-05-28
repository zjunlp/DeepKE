# coding=utf-8

import os
import re
import json
import jieba
from tqdm import tqdm


class Vocab(object):
    def __init__(
            self,
            min_count=2,
            data_path='data/origin/train.txt',
            vocab_path='data/word_vocab.json',
    ):
        self.min_count = min_count
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.word2id = self.load_vocab()

    def load_vocab(self):
        if os.path.exists(self.vocab_path):
            word2id = json.load(open(self.vocab_path, encoding='utf-8'))
            return word2id

        # otherwise read every word in the source files and build vocab
        origin_text = []
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                data = json.loads(str(line))

                # 每个 instance 为一个句子，用来处理。
                # 处理前先把标点符号全部去掉。
                instance = data['text'].strip()
                punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
                re_punc = "[{}]+".format(punc)
                instance = re.sub(re_punc, "", instance).strip()

                # 为了保证保证 entity 被独立分割为一个 word，会用一些 trick
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

                # 此时每个 ins 为用 list 格式保存的句子，每一项为 word
                # print(ins)
                origin_text.append(ins)

        words = {}
        # words counts
        for d in tqdm(iter(origin_text)):
            for w in d:
                words[w] = words.get(w, 0) + 1
        # filter by min_count
        words = {i: j for i, j in words.items() if j >= self.min_count}
        # 0: <pad>
        # 1: <mask>
        id2word = {i + 2: j for i, j in enumerate(words)}
        word2id = {j: i for i, j in id2word.items()}
        json.dump(word2id,
                  open(self.vocab_path, 'w', encoding='utf-8'),
                  ensure_ascii=False)
        return word2id


if __name__ == "__main__":
    v = Vocab(
        min_count=2,
        data_path='data/origin/train.txt',
        vocab_path='data/word_vocab.json',
    )
