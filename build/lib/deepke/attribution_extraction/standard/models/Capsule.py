import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
from . import BasicModule
from module import Embedding, CNN
from module import Capsule as CapsuleLayer

from utils import seq_len_to_mask, to_one_hot


class Capsule(BasicModule):
    def __init__(self, cfg):
        super(Capsule, self).__init__()

        if cfg.dim_strategy == 'cat':
            cfg.in_channels = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.in_channels = cfg.word_dim

        # capsule config
        cfg.input_dim_capsule = cfg.out_channels
        cfg.num_capsule = cfg.num_attributes

        self.num_attributes = cfg.num_attributes
        self.embedding = Embedding(cfg)
        self.cnn = CNN(cfg)
        self.capsule = CapsuleLayer(cfg)

    def forward(self, x):
        word, lens, entity_pos, attribute_value_pos = x['word'], x['lens'], x['entity_pos'], x['attribute_value_pos']
        mask = seq_len_to_mask(lens)
        inputs = self.embedding(word, entity_pos, attribute_value_pos)

        primary, _ = self.cnn(inputs)  # 由于长度改变，无法定向mask，不mask可可以，毕竟primary capsule 就是粗粒度的信息
        output = self.capsule(primary)
        output = output.norm(p=2, dim=-1)  # 求得模长再返回值

        return output  # [B, N]

    def loss(self, predict, target, reduction='mean'):
        m_plus, m_minus, loss_lambda = 0.9, 0.1, 0.5

        target = to_one_hot(target, self.num_attributes)
        max_l = (torch.relu(m_plus - predict))**2
        max_r = (torch.relu(predict - m_minus))**2
        loss = target * max_l + loss_lambda * (1 - target) * max_r
        loss = torch.sum(loss, dim=-1)

        if reduction == 'sum':
            return loss.sum()
        else:
            # 默认情况为求平均
            return loss.mean()
