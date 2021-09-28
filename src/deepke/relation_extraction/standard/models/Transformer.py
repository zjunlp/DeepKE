import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch.nn as nn
from . import BasicModule
from module import Embedding
from module import Transformer as TransformerBlock
from utils import seq_len_to_mask


class Transformer(BasicModule):
    def __init__(self, cfg):
        super(Transformer, self).__init__()

        if cfg.dim_strategy == 'cat':
            cfg.hidden_size = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.hidden_size = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.transformer = TransformerBlock(cfg)
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_relations)

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        mask = seq_len_to_mask(lens)
        inputs = self.embedding(word, head_pos, tail_pos)
        last_layer_hidden_state, all_hidden_states, all_attentions = self.transformer(inputs, key_padding_mask=mask)
        out_pool = last_layer_hidden_state.max(dim=1)[0]
        output = self.fc(out_pool)

        return output
