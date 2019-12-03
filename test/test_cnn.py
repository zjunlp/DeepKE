import pytest
import torch
from module import CNN
from utils import seq_len_to_mask


class Config(object):
    in_channels = 100
    out_channels = 200
    kernel_sizes = [3, 5, 7, 9, 11]
    activation = 'gelu'
    pooling_strategy = 'avg'


config = Config()


def test_CNN():

    x = torch.randn(4, 5, 100)
    seq = torch.arange(4, 0, -1)
    mask = seq_len_to_mask(seq, max_len=5)

    cnn = CNN(config)
    out, out_pooling = cnn(x, mask=mask)
    out_channels = config.out_channels * len(config.kernel_sizes)
    assert out.shape == torch.Size([4, 5, out_channels])
    assert out_pooling.shape == torch.Size([4, out_channels])


if __name__ == '__main__':
    pytest.main()
