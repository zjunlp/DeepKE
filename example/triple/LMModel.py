import os
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from transformers import BertModel

from utils import seq_len_to_mask


class LM(nn.Module):
    def __init__(self, cfg):
        super(LM, self).__init__()
        self.bert = BertModel.from_pretrained(cfg.lm_file, num_hidden_layers=cfg.num_hidden_layers)
        self.bilstm = RNN(cfg)
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.dropout)

    def load(self, path, device):
        
        self.load_state_dict(torch.load(path, map_location=device))


    def save(self, epoch=0, cfg=None):
        
        time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S')
        prefix = os.path.join(cfg.cwd, 'checkpoints',time_prefix)
        os.makedirs(prefix, exist_ok=True)
        name = os.path.join(prefix, cfg.model_name + '_' + f'epoch{epoch}' + '.pth')

        torch.save(self.state_dict(), name)
        return name

    def forward(self, x):
        word, lens = x['word'], x['lens']
        mask = seq_len_to_mask(lens, mask_pos_to_true=False)
        a = self.bert(word, attention_mask=mask)
        last_hidden_state = a[0]
        _, out_pool = self.bilstm(last_hidden_state, lens)
        out_pool = self.dropout(out_pool)
        output = self.fc(out_pool)

        return output


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()

        # self.xxx = config.xxx
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size // 2 if config.bidirectional else config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.last_layer_hn = config.last_layer_hn
        self.type_rnn = config.type_rnn

        rnn = eval(f'nn.{self.type_rnn}')
        self.rnn = rnn(input_size=self.input_size,
                       hidden_size=self.hidden_size,
                       num_layers=self.num_layers,
                       dropout=self.dropout,
                       bidirectional=self.bidirectional,
                       bias=True,
                       batch_first=True)

    
    def forward(self, x, x_len):
        B, L, _ = x.size()
        H, N = self.hidden_size, self.num_layers

        x_len = x_len.cpu()
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=True)
        output, hn = self.rnn(x)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=L)

        if self.type_rnn == 'LSTM':
            hn = hn[0]
        if self.bidirectional:
            hn = hn.view(N, 2, B, H).transpose(1, 2).contiguous().view(N, B, 2 * H).transpose(0, 1)
        else:
            hn = hn.transpose(0, 1)
        if self.last_layer_hn:
            hn = hn[:, -1, :]

        return output, hn

