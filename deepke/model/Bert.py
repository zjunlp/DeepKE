import torch.nn as nn
from deepke.model import BasicModule
from pytorch_transformers import BertModel


class Bert(BasicModule):
    def __init__(self, vocab_size, config):
        super(Bert, self).__init__()
        self.model_name = 'Bert'
        self.lm_name = config.lm_name
        self.out_dim = config.relation_type

        self.lm = BertModel.from_pretrained(self.lm_name)
        self.fc = nn.Linear(768, self.out_dim)

    def forward(self, x):
        out = self.lm(x)[-1]
        out = self.fc(out)
        return out
