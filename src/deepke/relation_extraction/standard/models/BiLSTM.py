import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch.nn as nn
from . import BasicModule
from module import Embedding, RNN


class BiLSTM(BasicModule):
    def __init__(self, cfg):
        super(BiLSTM, self).__init__()

        if cfg.dim_strategy == 'cat':
            cfg.input_size = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.input_size = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.bilstm = RNN(cfg)
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        inputs = self.embedding(word, head_pos, tail_pos)
        out, out_pool = self.bilstm(inputs, lens)
        output = self.fc(out_pool)

        return output
