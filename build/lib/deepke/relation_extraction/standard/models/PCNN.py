import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BasicModule
from module import Embedding, CNN
from utils import seq_len_to_mask


class PCNN(BasicModule):
    def __init__(self, cfg):
        super(PCNN, self).__init__()

        self.use_pcnn = cfg.use_pcnn
        if cfg.dim_strategy == 'cat':
            cfg.in_channels = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.in_channels = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.cnn = CNN(cfg)
        self.fc1 = nn.Linear(len(cfg.kernel_sizes) * cfg.out_channels, cfg.intermediate)
        self.fc2 = nn.Linear(cfg.intermediate, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.dropout)

        if self.use_pcnn:
            self.fc_pcnn = nn.Linear(3 * len(cfg.kernel_sizes) * cfg.out_channels,
                                     len(cfg.kernel_sizes) * cfg.out_channels)
            self.pcnn_mask_embedding = nn.Embedding(4, 3)
            masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
            self.pcnn_mask_embedding.weight.data.copy_(masks)
            self.pcnn_mask_embedding.weight.requires_grad = False


    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        mask = seq_len_to_mask(lens)

        inputs = self.embedding(word, head_pos, tail_pos)
        out, out_pool = self.cnn(inputs, mask=mask)

        if self.use_pcnn:
            out = out.unsqueeze(-1)  # [B, L, Hs, 1]
            pcnn_mask = x['pcnn_mask']
            pcnn_mask = self.pcnn_mask_embedding(pcnn_mask).unsqueeze(-2)  # [B, L, 1, 3]
            out = out + pcnn_mask  # [B, L, Hs, 3]
            out = out.max(dim=1)[0] - 100  # [B, Hs, 3]
            out_pool = out.view(out.size(0), -1)  # [B, 3 * Hs]
            out_pool = F.leaky_relu(self.fc_pcnn(out_pool))  # [B, Hs]
            out_pool = self.dropout(out_pool)

        output = self.fc1(out_pool)
        output = F.leaky_relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output
