# coding=utf-8

import warnings
import json

class Config(object):
    # reproducibility
    seed = 1

    # path
    train_data_path = 'data/origin/train.txt'
    test_data_path = 'data/origin/test.txt'
    vocab_path = 'data/word_vocab.json'
    save_path = 'checkpoints'
    load_path = 'checkpoints/PCNN_ATT.pkl'

    # vocab
    min_count = 2

    # model hyterparams
    embedding_size = 200
    hidden_size = 300
    output_size = 57

    # train epoch
    epochs = 10
    batch_size = 32
    lr = 3e-4

    # gpu
    gpu = 0

    def _parse(self, kwargs):
        """
        根据字典 kwargs 更新 config 参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


config = Config()






relations = [
 ('无关系', 3557),
 ('总工程师', 38),
 ('首席执行官', 177),
 ('财务总监', 947),
 ('总裁助理', 19),
 ('首席风险官', 22),
 ('非独立董事', 115),
 ('首席运营执行官', 3),
 ('独立非执行董事', 4),
 ('名誉董事', 7),
 ('首席技术官', 7),
 ('董事长', 15292),
 ('总裁', 2011),
 ('执行董事长', 14),
 ('副首席执行官', 13),
 ('行长', 269),
 ('首席投资执行官', 11),
 ('常务副总裁', 16),
 ('总经济师', 4),
 ('监事', 204),
 ('董事会秘书', 1062),
 ('法定代表人', 274),
 ('监事长', 63),
 ('非执行董事', 81),
 ('董事会办公室主任', 2),
 ('高级副总裁', 57),
 ('名誉董事长', 8),
 ('董事局主席', 16),
 ('联席总裁', 13),
 ('副经理', 6),
 ('执行总裁', 47),
 ('监事会主席', 85),
 ('人事行政总监', 11),
 ('首席知识官', 5),
 ('副行长', 273),
 ('副董事长', 914),
 ('独立董事', 272),
 ('副总裁', 1140),
 ('总经理助理', 7),
 ('股东代表监事', 6),
 ('职工代表监事', 13),
 ('副总经理', 218),
 ('财务负责人', 55),
 ('职工监事', 19),
 ('董事', 2528),
 ('总会计师', 21),
 ('非职工监事', 7),
 ('总精算师', 13),
 ('执行董事', 377),
 ('合规总监', 3),
 ('总经理', 3131),
 ('常务副总经理', 73),
 ('首席财务官', 27),
 ('执行副总裁', 32),
 ('外部监事', 4),
 ('行长助理', 5),
 ('助理总裁', 29)]