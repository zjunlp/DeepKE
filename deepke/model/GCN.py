import torch
import torch.nn as nn
import torch.nn.functional as F
from deepke.model import BasicModule, Embedding


class GCN(BasicModule):
    def __init__(self, vocab_size, config):
        super(GCN, self).__init__()
        self.model_name = 'GCN'
        # TODO
        # 暂时有bug，主要是没有找到很好的可以做中文 dependency parsing 的工具
