import pytest
import torch
from module import Transformer
from utils import seq_len_to_mask


class Config():
    hidden_size = 12
    intermediate_size = 24
    num_hidden_layers = 5
    num_heads = 3
    dropout = 0.0
    layer_norm_eps = 1e-12
    hidden_act = 'gelu_new'
    output_attentions = True
    output_hidden_states = True


config = Config()


def test_Transformer():
    m = Transformer(config)
    i = torch.randn(4, 5, 12)  # [B, L, H]
    key_padding_mask = seq_len_to_mask([5, 4, 3, 2], max_len=5)
    attention_mask = torch.tensor([1, 0, 0, 1, 0])  # 为1 的地方 mask 掉
    head_mask = torch.tensor([0, 1, 0])  # 为1 的地方 mask 掉

    out = m(i, key_padding_mask=key_padding_mask, attention_mask=attention_mask, head_mask=head_mask)
    hn, h_all, att_weights = out
    assert hn.shape == torch.Size([4, 5, 12])
    assert torch.equal(h_all[0], i) and torch.equal(h_all[-1], hn) == True
    assert len(h_all) == config.num_hidden_layers + 1
    assert len(att_weights) == config.num_hidden_layers
    assert att_weights[0].shape == torch.Size([4, 3, 5, 5])
    assert att_weights[0].unbind(dim=1)[1].bool().any() == False


if __name__ == '__main__':
    pytest.main()
