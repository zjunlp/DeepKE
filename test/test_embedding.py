import pytest
import torch
from module import Embedding


class Config(object):
    vocab_size = 10
    word_dim = 10
    pos_size = 12  # 2 * pos_limit + 2
    pos_dim = 5
    dim_strategy = 'cat'  # [cat, sum]


config = Config()

x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 3, 5, 0], [8, 4, 3, 0, 0]])
x_pos = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]])


def test_Embedding_cat():
    embed = Embedding(config)
    feature = embed((x, x_pos))
    dim = config.word_dim + config.pos_dim

    assert feature.shape == torch.Size((3, 5, dim))


def test_Embedding_sum():
    config.dim_strategy = 'sum'
    embed = Embedding(config)
    feature = embed((x, x_pos))
    dim = config.word_dim

    assert feature.shape == torch.Size((3, 5, dim))


if __name__ == '__main__':
    pytest.main()
