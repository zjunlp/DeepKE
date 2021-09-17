import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DotAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super(DotAttention, self).__init__()
        self.dropout = dropout

    def forward(self, Q, K, V, mask_out=None, head_mask=None):
        """
        一般输入信息 X 时，假设 K = V = X

        att_weight = softmax( score_func(q, k) )
        att = sum( att_weight * v )

        :param Q: [..., L, H]
        :param K: [..., S, H]
        :param V: [..., S, H]
        :param mask_out: [..., 1, S]
        :return:
        """
        H = Q.size(-1)

        scale = float(H)**0.5
        attention_weight = torch.matmul(Q, K.transpose(-1, -2)) / scale

        if mask_out is not None:
            # 当 DotAttention 单独使用时（几乎不会），保证维度一样
            while mask_out.dim() != Q.dim():
                mask_out = mask_out.unsqueeze(1)
            attention_weight.masked_fill_(mask_out, -1e8)

        attention_weight = F.softmax(attention_weight, dim=-1)

        attention_weight = F.dropout(attention_weight, self.dropout)

        # mask heads if we want to:
        # multi head 才会使用
        if head_mask is not None:
            attention_weight = attention_weight * head_mask

        attention_out = torch.matmul(attention_weight, V)

        return attention_out, attention_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, output_attentions=True):
        """
        :param embed_dim: 输入的维度，必须能被 num_heads 整除
        :param num_heads: attention 的个数
        :param dropout: float。
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.output_attentions = output_attentions
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        assert self.all_head_dim == embed_dim, logger.error(
            f"embed_dim{embed_dim} must be divisible by num_heads{num_heads}")

        self.q_in = nn.Linear(embed_dim, self.all_head_dim)
        self.k_in = nn.Linear(embed_dim, self.all_head_dim)
        self.v_in = nn.Linear(embed_dim, self.all_head_dim)
        self.attention = DotAttention(dropout=dropout)
        self.out = nn.Linear(self.all_head_dim, embed_dim)

    def forward(self, Q, K, V, key_padding_mask=None, attention_mask=None, head_mask=None):
        """
        :param Q: [B, L, Hs]
        :param K: [B, S, Hs]
        :param V: [B, S, Hs]
        :param key_padding_mask: [B, S]                为 1/True 的地方需要 mask
        :param attention_mask: [S] / [L, S] 指定位置 mask 掉， 为 1/True 的地方需要 mask
        :param head_mask: [N] 指定 head mask 掉，        为 1/True 的地方需要 mask
        """
        B, L, Hs = Q.shape
        S = V.size(1)
        N, H = self.num_heads, self.head_dim

        q = self.q_in(Q).view(B, L, N, H).transpose(1, 2)  # [B, N, L, H]
        k = self.k_in(K).view(B, S, N, H).transpose(1, 2)  # [B, N, S, H]
        v = self.v_in(V).view(B, S, N, H).transpose(1, 2)  # [B, N, S, H]

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.ne(0)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

        if attention_mask is not None:
            attention_mask = attention_mask.ne(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
            else:
                raise ValueError(f'attention_mask dim must be 1 or 2, can not be {attention_mask.dim()}')

        if key_padding_mask is None:
            mask_out = attention_mask if attention_mask is not None else None
        else:
            mask_out = (key_padding_mask + attention_mask).ne(0) if attention_mask is not None else key_padding_mask

        if head_mask is not None:
            head_mask = head_mask.eq(0)
            head_mask = head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        attention_out, attention_weight = self.attention(q, k, v, mask_out=mask_out, head_mask=head_mask)

        attention_out = attention_out.transpose(1, 2).reshape(B, L, N * H)  # [B, N, L, H] -> [B, L, N * H]

        # concat all heads, and do output linear
        attention_out = self.out(attention_out)  # [B, L, N * H] -> [B, L, H]

        if self.output_attentions:
            return attention_out, attention_weight
        else:
            return attention_out,


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from utils import seq_len_to_mask

    q = torch.randn(4, 6, 20)  # [B, L, H]
    k = v = torch.randn(4, 5, 20)  # [B, S, H]
    key_padding_mask = seq_len_to_mask([5, 4, 3, 2], max_len=5)
    attention_mask = torch.tensor([1, 0, 0, 1, 0])  # 为1 的地方 mask 掉
    head_mask = torch.tensor([0, 1])  # 为1 的地方 mask 掉

    m = MultiHeadAttention(embed_dim=20, num_heads=2, dropout=0.0, output_attentions=True)
    ao, aw = m(q, k, v, key_padding_mask=key_padding_mask, attention_mask=attention_mask, head_mask=head_mask)
    print(ao.shape, aw.shape)  # [B, L, H]  [B, N, L, S]
    print(ao)
    print(aw.unbind(1))
