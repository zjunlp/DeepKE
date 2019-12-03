import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BasicModule
from module import Embedding, RNN
from utils import seq_len_to_mask


class BiLSTM(BasicModule):
    def __init__(self, cfg):
        super(BiLSTM, self).__init__()

        self.use_pcnn = cfg.use_pcnn

        self.embedding = Embedding(cfg)
        self.bilsm = RNN(cfg)
        self.fc1 = nn.Linear(len(cfg.kernel_sizes) * cfg.out_channels, cfg.intermediate)
        self.fc2 = nn.Linear(cfg.intermediate, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        inputs = self.embedding(word, head_pos, tail_pos)
        out, out_pool = self.rnn(inputs)
