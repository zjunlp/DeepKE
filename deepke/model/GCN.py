import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from deepke.model import BasicModule, Embedding

# 暂时有bug，主要是没有找到很好的可以做中文 dependency parsing 的工具
# 尝试了 hanlp, standford_nlp, 都需要安装 java 包，还是老版本的 java6，测试时bug不少


class GCN(BasicModule):
    def __init__(self, vocab_size, config):
        super(GCN, self).__init__()
        self.model_name = 'GCN'
        self.vocab_size = vocab_size
        self.word_dim = config.model.word_dim
        self.pos_size = config.model.pos_size
        self.pos_dim = config.model.pos_dim
        self.hidden_dim = config.model.hidden_dim
        self.dropout = config.model.dropout
        self.num_layers = config.gcn.num_layers
        self.out_dim = config.relation_type
        self.embedding = Embedding(self.vocab_size, self.word_dim, self.pos_size, self.pos_dim)
        self.input_dim = self.word_dim + self.pos_dim * 2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input):
        *x, adj, mask = input
        x = self.embedding(x)
        for i in range(1, self.num_layers + 1):
            if i == 1 == self.num_layers:
                out = self.fc1(torch.bmm(adj, x))
            elif i == self.num_layers:
                out = self.fc3(torch.bmm(adj, x))
            else:
                out = F.relu(self.fc2(torch.bmm(adj, x)))
        return out


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        s = getattr(self, '_size', -1)
        if s != -1:
            return self._size
        else:
            count = 1
            for i in range(self.num_children):
                count += self.children[i].size()
            self._size = count
            return self._size

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

    def depth(self):
        d = getattr(self, '_depth', -1)
        if d != -1:
            return self._depth
        else:
            count = 0
            if self.num_children > 0:
                for i in range(self.num_children):
                    child_depth = self.children[i].depth()
                    if child_depth > count:
                        count = child_depth
                count += 1
            self._depth = count
            return self._depth


def head_to_adj(head, directed=True, self_loop=False):
    """
    Convert a sequence of head indexes to an (numpy) adjacency matrix.
    """
    seq_len = len(head)
    head = head[:seq_len]
    root = None
    nodes = [Tree() for _ in head]

    for i in range(seq_len):
        h = head[i]
        setattr(nodes[i], 'idx', i)
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None

    ret = np.zeros((seq_len, seq_len), dtype=np.float32)
    queue = [root]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        idx += [t.idx]
        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


if __name__ == '__main__':
    inputs = torch.tensor([list(range(6))])
    embedding = nn.Embedding(10, 10)
    inputs = embedding(inputs)

    head = [2, 0, 5, 3, 2, 2]
    adj = head_to_adj(head, directed=False, self_loop=True)
    print(adj)
    adj = torch.tensor([adj])

    model = GCN(10, 10)
    outs = model(adj, inputs)
    print(outs.shape)
