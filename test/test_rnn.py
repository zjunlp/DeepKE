import pytest
import torch
from module import RNN
from utils import seq_len_to_mask


class Config(object):
    type_rnn = 'LSTM'
    input_size = 5
    hidden_size = 4
    num_layers = 3
    dropout = 0.0
    last_layer_hn = False
    bidirectional = True


config = Config()


def test_CNN():
    torch.manual_seed(1)
    x = torch.tensor([[4, 3, 2, 1], [5, 6, 7, 0], [8, 10, 0, 0]])
    x = torch.nn.Embedding(11, 5, padding_idx=0)(x)  # B,L,H = 3,4,5
    x_len = torch.tensor([4, 3, 2])

    model = RNN(config)
    output, hn = model(x, x_len)

    B, L, _ = x.size()
    H, N = config.hidden_size, config.num_layers

    assert output.shape == torch.Size([B, L, H])
    assert hn.shape == torch.Size([B, N, H])

    config.bidirectional = False
    model = RNN(config)
    output, hn = model(x, x_len)
    assert output.shape == torch.Size([B, L, H])
    assert hn.shape == torch.Size([B, N, H])

    config.last_layer_hn = True
    model = RNN(config)
    output, hn = model(x, x_len)
    assert output.shape == torch.Size([B, L, H])
    assert hn.shape == torch.Size([B, H])


if __name__ == '__main__':
    pytest.main()
