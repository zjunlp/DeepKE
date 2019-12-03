import pytest
import torch
from utils import seq_len_to_mask
from module import DotAttention, MultiHeadAttention

torch.manual_seed(1)
q = torch.randn(4, 6, 20)  # [B, L, H]
k = v = torch.randn(4, 5, 20)  # [B, S, H]
key_padding_mask = seq_len_to_mask([5, 4, 3, 2], max_len=5)
attention_mask = torch.tensor([1, 0, 0, 1, 0])  # 为1 的地方 mask 掉
head_mask = torch.tensor([0, 1, 0, 0])  # 为1 的地方 mask 掉

# m = DotAttention(dropout=0.0)
# ao,aw = m(q,k,v,key_padding_mask)
# print(ao.shape,aw.shape)
# print(aw)


def test_DotAttention():
    m = DotAttention(dropout=0.0)
    ao, aw = m(q, k, v, mask_out=key_padding_mask)

    assert ao.shape == torch.Size([4, 6, 20])
    assert aw.shape == torch.Size([4, 6, 5])
    assert torch.all(aw[1, :, -1:].eq(0)) == torch.all(aw[2, :, -2:].eq(0)) == torch.all(aw[3, :, -3:].eq(0)) == True


def test_MultiHeadAttention():
    m = MultiHeadAttention(embed_dim=20, num_heads=4, dropout=0.0)
    ao, aw = m(q, k, v, key_padding_mask=key_padding_mask,attention_mask=attention_mask,head_mask=head_mask)

    assert ao.shape == torch.Size([4, 6, 20])
    assert aw.shape == torch.Size([4, 4, 6, 5])
    assert aw.unbind(dim=1)[1].bool().any() == False


if __name__ == '__main__':
    pytest.main()
