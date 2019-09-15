import torch
import torch.nn as nn
import torch.nn.functional as F
from deepke.model import BasicModule, Embedding


class CNN(BasicModule):
    def __init__(self, vocab_size, config):
        super(CNN, self).__init__()
        self.model_name = 'CNN'
        # TODO