import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim, hidden_size, drop_out, bidirectional, num_layers):
        super().__init__()
        """ nn.Embedding: parameter size (num_words, embedding_dim) """
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            dropout = drop_out,
            num_layers = num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(hidden_size * 2, num_labels)\

        """https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html"""
        self.crf = CRF(num_labels, batch_first=True)

    def _get_lstm_feature(self, input):
        # out = self.embed(input)
        # packed = pack_padded_sequence(out, lengths.cpu(), batch_first=True)
        # out, _ = self.lstm(packed)
        # out, _ = pad_packed_sequence(out, batch_first=True)
        # return self.linear(out)
        out = self.embed(input)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask)
        # return -self.crf.forward(y_pred, target, mask, reduction='mean')