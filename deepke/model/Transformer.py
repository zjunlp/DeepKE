import math
import torch
import torch.nn as nn
from deepke.model import BasicModule, Embedding


class DotAttention(nn.Module):
    '''
    \text {Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
    '''
    def __init__(self, dropout=0.0):
        super(DotAttention, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask_out=None):
        """
        :param Q: [batch, seq_len_q, feature_size]
        :param K: [batch, seq_len_k, feature_size]
        :param V: [batch, seq_len_k, feature_size]
        :param mask_out: [batch, 1, seq_len] or [batch, seq_len_q, seq_len_k]
        """
        feature_size = Q.size(-1)
        scale = math.sqrt(feature_size)
        output = torch.matmul(Q, K.transpose(1, 2)) / scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -1e18)
        output = self.softmax(output)
        output = self.drop(output)
        return torch.matmul(output, V)


class MultiHeadAttention(nn.Module):
    """
    :param feature_size: int, 输入维度的大小。同时也是输出维度的大小。
    :param num_head: int，head的数量。
    :param dropout: float。
    """
    def __init__(self, feature_size, num_head, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.feature_size = feature_size
        self.num_head = num_head
        self.q_in = nn.Linear(feature_size, feature_size * num_head)
        self.k_in = nn.Linear(feature_size, feature_size * num_head)
        self.v_in = nn.Linear(feature_size, feature_size * num_head)
        self.attention = DotAttention(dropout=dropout)
        self.out = nn.Linear(feature_size * num_head, feature_size)

    def forward(self, Q, K, V, att_mask_out=None):
        """
        :param Q: [batch, seq_len_q, feature_size]
        :param K: [batch, seq_len_k, feature_size]
        :param V: [batch, seq_len_k, feature_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, sq, feature = Q.size()
        sk = K.size(1)
        n_head = self.num_head
        # input linear
        q = self.q_in(Q).view(batch, sq, n_head, feature)
        k = self.k_in(K).view(batch, sk, n_head, feature)
        v = self.v_in(V).view(batch, sk, n_head, feature)

        # transpose q, k and v to do batch attention
        # [batch, seq_len, num_head, feature] => [num_head*batch, seq_len, feature]
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, sq, feature)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, sk, feature)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, sk, feature)
        if att_mask_out is not None:
            att_mask_out = att_mask_out.repeat(n_head, 1, 1)
        att = self.attention(q, k, v,
                             att_mask_out).view(n_head, batch, sq, feature)

        # concat all heads, do output linear
        # [num_head, batch, seq_len, feature] => [batch, seq_len, num_head*feature]
        att = att.permute(1, 2, 0, 3).contiguous().view(batch, sq, -1)
        output = self.out(att)
        return output


class Transformer(BasicModule):
    def __init__(self, vocab_size, config):
        super(Transformer, self).__init__()
        self.model_name = 'Transformer'
        self.word_dim = config.word_dim
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.out_dim = config.relation_type
        self.layers = config.transformer_layers

        self.embedding = Embedding(vocab_size, self.word_dim, self.pos_size,
                                   self.pos_dim)
        self.feature_dim = self.word_dim + self.pos_dim * 2
        self.att = MultiHeadAttention(self.feature_dim, num_head=4)
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.ffn = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.feature_dim),
                                 nn.Dropout(self.dropout))
        self.norm2 = nn.LayerNorm(self.feature_dim)
        self.fc = nn.Linear(self.feature_dim, self.out_dim)

    def forward(self, input):
        *x, mask = input
        x = self.embedding(x)
        att_mask_out = mask.eq(0).unsqueeze(1)

        for i in range(self.layers):
            attention = self.att(x, x, x, att_mask_out)
            norm_att = self.norm1(attention + x)
            x = self.ffn(norm_att)
            x = self.norm2(x + norm_att)
        x = x[:, 0]
        out = self.fc(x)
        return out


if __name__ == '__main__':
    torch.manual_seed(1)

    q = torch.randn(32, 50, 100)
    k = torch.randn(32, 60, 100)
    v = torch.randn(32, 60, 100)
    mask = torch.randn(32, 60).unsqueeze(1).gt(0)

    att1 = DotAttention()
    out = att1(q, k, v, mask)
    print(out.shape)  # [32, 50, 100]

    att2 = MultiHeadAttention(feature_size=100, num_head=8)
    out = att2(q, k, v, mask)
    print(out.shape)  # [32, 50, 100]
