# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .LSTM4VarLenSeq import LSTM4VarLenSeq
from config import config
from utils import set_seed

set_seed(config.seed)


class BiLSTM_ATT(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_size, hidden_size,
                 output_size):
        super(BiLSTM_ATT, self).__init__()
        self.model_name = 'BiLSTM_ATT'
        self.num_directions = 2
        self.word_embedding = nn.Embedding(word_vocab_size,
                                           word_embedding_size,
                                           padding_idx=0)
        self.init_word_embedding()

        self.lstm = LSTM4VarLenSeq(word_embedding_size,
                                   hidden_size,
                                   bidirectional=True,
                                   init='orthogonal',
                                   take_last=False)

        # Note that we use bi-directional hidden states.
        self.batch = config.batch_size
        self.att_weight = nn.Parameter(
            torch.randn( 1, hidden_size * 2))
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def attention(self, H):
        H = torch.transpose(H, 1, 2)
        M = H.tanh()
        a = F.softmax(torch.bmm(self.att_weight, M), 2)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)  # batch * (hidden * 2) * 1

    def forward(self, x, x_len):
        x_embed = self.word_embedding(x)
        # B, L, H * num_directions
        lstm_out, (hn, cn) = self.lstm(x_embed, x_len)

        att_out = self.attention(lstm_out).tanh()
        att_out = att_out.squeeze(2)

        fc_out = F.leaky_relu(self.fc(att_out), negative_slope=0.1)
        return fc_out

    def init_word_embedding(self):
        self.word_embedding.weight.data.requires_grad = True
        nn.init.xavier_normal_(self.word_embedding.weight)
        # padding idx defaults to 0.
        with torch.no_grad():
            self.word_embedding.weight[0].fill_(0)


if __name__ == '__main__':
    # Unit test for LSTM variable length sequences
    # ================
    inputs = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0], [2, 4, 3, 0],
                           [1, 4, 3, 0], [1, 2, 3, 4]])
    lens = torch.LongTensor([3, 2, 3, 3, 4])
    net = BiLSTM_ATT(100, 200, 300)
    net(inputs, lens)
