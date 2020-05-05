import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, config):
        """
        word embedding: 一般 0 为 padding
        pos embedding:  一般 0 为 padding
        dim_strategy: [cat, sum]  多个 embedding 是拼接还是相加
        """
        super(Embedding, self).__init__()

        # self.xxx = config.xxx
        self.vocab_size = config.vocab_size
        self.word_dim = config.word_dim
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim if config.dim_strategy == 'cat' else config.word_dim
        self.dim_strategy = config.dim_strategy

        self.wordEmbed = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=0)
        self.headPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        self.tailPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        
        self.layer_norm = nn.LayerNorm(self.word_dim)

    def forward(self, *x):
        word, head, tail = x
        word_embedding = self.wordEmbed(word)
        head_embedding = self.headPosEmbed(head)
        tail_embedding = self.tailPosEmbed(tail)

        if self.dim_strategy == 'cat':
            return torch.cat((word_embedding, head_embedding, tail_embedding), -1)
        elif self.dim_strategy == 'sum':
            # 此时 pos_dim == word_dim
            return self.layer_norm(word_embedding + head_embedding + tail_embedding)
        else:
            raise Exception('dim_strategy must choose from [sum, cat]')
