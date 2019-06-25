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
