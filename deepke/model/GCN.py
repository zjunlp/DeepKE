import torch
import torch.nn as nn
import torch.nn.functional as F
from deepke.model import BasicModule, Embedding


class GCN(BasicModule):
    def __init__(self, vocab_size, config):
        super(GCN, self).__init__()
        self.model_name = 'GCN'
        # TODO