import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)



class GCN(nn.Module):
    def __init__(self,cfg):
        super(GCN , self).__init__()

        self.num_layers = cfg.num_layers
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.dropout = cfg.dropout

        self.fc1 = nn.Linear(self.input_size , self.hidden_size)
        self.fc = nn.Linear(self.hidden_size , self.hidden_size)
        self.weight_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.weight_list.append(nn.Linear(self.hidden_size * (i + 1),self.hidden_size))
        self.dropout = nn.Dropout(self.dropout)

    def forward(self , x, adj):
        L = adj.sum(2).unsqueeze(2) + 1
        outputs = self.fc1(x)
        cache_list = [outputs]
        output_list = []
        for l in range(self.num_layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)
            AxW = AxW / L
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list , dim=2)
            output_list.append(self.dropout(gAxW))
        # gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = output_list[self.num_layers - 1]
        gcn_outputs = gcn_outputs + self.fc1(x)

        out = self.fc(gcn_outputs)
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


def pad_adj(adj, max_len):
    pad_len = max_len - adj.shape[0]
    for i in range(pad_len):
        adj = np.insert(adj, adj.shape[-1], 0, axis=1)
    for i in range(len):
        adj = np.insert(adj, adj.shape[0], 0, axis=0)

