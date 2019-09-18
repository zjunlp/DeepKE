import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from deepke.model import BasicModule, Embedding


class VarLenLSTM(BasicModule):
    def __init__(self, input_size, hidden_size, lstm_layers=1, dropout=0, last_hn=False):
        super(VarLenLSTM, self).__init__()
        self.model_name = 'VarLenLSTM'
        self.lstm_layers = lstm_layers
        self.last_hn = last_hn
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True,
            bias=True,
            batch_first=True,
        )

    def forward(self, x, x_len):
        '''
        针对有 padding 的句子
        一般来说，out 用来做序列标注，hn 做分类任务
        :param x:      [B * L * H]
        :param x_len:  [l...]
        :return:
            out:  [B * seq_len * hidden]   hidden = 2 * hidden_dim
             hn:  [B * layers  * hidden]   hidden = 2 * hidden_dim
        '''
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=True)
        out, (hn, _) = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=0.0)
        hn = hn.transpose(0, 1).contiguous()
        # [B, layers, 2*hidden]
        hn = hn.view(hn.size(0), self.lstm_layers, -1)
        if self.last_hn:
            hn = hn[:, -1].unsqueeze(1)

        return out, hn


class BiLSTM(BasicModule):
    def __init__(self, vocab_size, config):
        super(BiLSTM, self).__init__()
        self.model_name = 'BiLSTM'
        self.vocab_size = vocab_size
        self.word_dim = config.model.word_dim
        self.pos_size = config.model.pos_size
        self.pos_dim = config.model.pos_dim
        self.hidden_dim = config.model.hidden_dim
        self.dropout = config.model.dropout
        self.lstm_layers = config.rnn.lstm_layers
        self.last_hn = config.rnn.last_hn
        self.out_dim = config.relation_type

        self.embedding = Embedding(self.vocab_size, self.word_dim, self.pos_size, self.pos_dim)
        self.input_dim = self.word_dim + self.pos_dim * 2
        self.lstm = VarLenLSTM(self.input_dim,
                               self.hidden_dim,
                               self.lstm_layers,
                               dropout=self.dropout,
                               last_hn=self.last_hn)
        if self.last_hn:
            linear_input_dim = self.hidden_dim * 2
        else:
            linear_input_dim = self.hidden_dim * 2 * self.lstm_layers
        self.fc1 = nn.Linear(linear_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, input):
        *x, mask = input
        x = self.embedding(x)
        x_lens = torch.sum(mask.gt(0), dim=-1)
        _, hn = self.lstm(x, x_lens)
        hn = hn.view(hn.size(0), -1)
        y = F.leaky_relu(self.fc1(hn))
        y = F.leaky_relu(self.fc2(y))
        return y


if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.Tensor([
        [1, 2, 3, 4, 3, 2],
        [1, 2, 3, 0, 0, 0],
        [2, 4, 3, 0, 0, 0],
        [2, 3, 0, 0, 0, 0],
    ])
    x_len = torch.Tensor([6, 3, 3, 2])
    embedding = nn.Embedding(5, 10, padding_idx=0)
    model = VarLenLSTM(input_size=10, hidden_size=30, lstm_layers=5, last_hn=False)

    x = embedding(x)  # [4, 6, 5]
    out, hn = model(x, x_len)
    # out: [4, 6, 60]   [B, seq_len, 2 * hidden]
    #  hn: [4, 5, 60]   [B, layers,  2 * hidden]
    print(out.shape, hn.shape)
