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

