import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GCN(nn.Module):
    def __init__(self, cfg):
        super(GCN, self).__init__()

        # self.xxx = config.xxx
        self.num_layers = cfg.num_layers
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.dropout = cfg.dropout

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers - 1)])
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x, adj):
        L = x.size(1)
        AxW = self.fc1(torch.bmm(adj, x)) + self.fc1(x)
        AxW = AxW / L
        AxW = F.leaky_relu(AxW)
        AxW = self.dropout(AxW)
        for fc in self.fcs:
            AxW = fc(torch.bmm(adj, AxW)) + fc(AxW)
            AxW = AxW / L
            AxW = F.leaky_relu(AxW)
            AxW = self.dropout(AxW)

        return AxW





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


def pad_adj(adj, max_len):
    pad_len = max_len - adj.shape[0]
    for i in range(pad_len):
        adj = np.insert(adj, adj.shape[-1], 0, axis=1)
    for i in range(len):
        adj = np.insert(adj, adj.shape[0], 0, axis=0)



if __name__ == '__main__':
    class Config():
        num_layers = 3
        input_size = 50
        hidden_size = 100
        dropout = 0.3
    cfg = Config()
    x = torch.randn(1, 10, 50)
    adj = torch.empty(1, 10, 10).random_(2)
    m = GCN(cfg)
    print(m)
    out = m(x, adj)
    print(out.shape)
    print(out)