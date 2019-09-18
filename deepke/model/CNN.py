import torch
import torch.nn as nn
import torch.nn.functional as F
from deepke.model import BasicModule, Embedding


class CNN(BasicModule):
    def __init__(self, vocab_size, config):
        super(CNN, self).__init__()
        self.model_name = 'CNN'
        self.vocab_size = vocab_size
        self.word_dim = config.model.word_dim
        self.pos_size = config.model.pos_size
        self.pos_dim = config.model.pos_dim
        self.hidden_dim = config.model.hidden_dim
        self.dropout = config.model.dropout
        self.use_pcnn = config.cnn.use_pcnn
        self.out_channels = config.cnn.out_channels
        self.kernel_size = config.cnn.kernel_size
        self.out_dim = config.relation_type

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]
        for k in self.kernel_size:
            assert k % 2 == 1, "kernel size has to be odd numbers."

        self.embedding = Embedding(self.vocab_size, self.word_dim, self.pos_size, self.pos_dim)
        # PCNN embedding
        self.mask_embed = nn.Embedding(4, 3)
        masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
        self.mask_embed.weight.data.copy_(masks)
        self.mask_embed.weight.requires_grad = False

        self.input_dim = self.word_dim + self.pos_dim * 2
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.input_dim,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k // 2,
                      bias=None) for k in self.kernel_size
        ])
        self.conv_dim = len(self.kernel_size) * self.out_channels
        if self.use_pcnn:
            self.conv_dim *= 3
        self.fc1 = nn.Linear(self.conv_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input):
        *x, mask = input
        x = self.embedding(x)
        mask_embed = self.mask_embed(mask)

        # [B,L,C] -> [B,C,L]
        x = torch.transpose(x, 1, 2)

        # CNN
        x = [F.leaky_relu(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)

        # mask
        mask = mask.unsqueeze(1)  # B x 1 x L
        x = x.masked_fill_(mask.eq(0), float('-inf'))

        if self.use_pcnn:
            # triple max_pooling
            x = x.unsqueeze(-1).permute(0, 2, 1, 3)  # [B, L, C, 1]
            mask_embed = mask_embed.unsqueeze(-2)  # [B, L, 1, 3]
            x = x + mask_embed  # [B, L, C, 3]
            x = torch.max(x, dim=1)[0] - 100  # [B, C, 3]
            x = x.view(x.size(0), -1)  # [B, 3*C]

        else:
            # max_pooling
            x = F.max_pool1d(x, x.size(-1)).squeeze(-1)  # [[B,C],..]

        # droup
        x = self.dropout(x)

        # linear
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        return x


if __name__ == '__main__':
    pass
