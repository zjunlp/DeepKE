import torch
import torch.nn as nn
import torch.nn.functional as F
from .LSTM4VarLenSeq import LSTM4VarLenSeq


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.name = BiLSTM
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size,
                                      padding_idx=1)
        self.init_embedding()
        self.bilstm = LSTM4VarLenSeq(embedding_size,
                                     hidden_size,
                                     num_layers=1,
                                     bidirectional=True,
                                     take_last=True)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def init_embedding(self):
        self.embedding.weight.data.requires_grad = True
        nn.init.xavier_normal_(self.embedding.weight)
        # padding idx defaults to 0.
        with torch.no_grad():
            self.embedding.weight[1].fill_(0)

    def forward(self, text, lengths):
        inputs = self.embedding(text)
        # L, B, H * num_directions
        outputs = self.bilstm(inputs, lengths)
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(outputs.size(0), -1)
        outputs = F.leaky_relu(self.fc1(outputs), negative_slope=0.1)
        outputs = self.fc2(outputs)
        return outputs
