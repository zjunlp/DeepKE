import torch.nn as nn
from deepke.model import BasicModule
from pytorch_transformers import BertModel


class LM(BasicModule):
    def __init__(self, vocab_size, config):
        super(LM, self).__init__()
        self.model_name = 'LM'
        self.lm_name = config.lm.lm_file
        self.out_dim = config.relation_type

        self.lm = BertModel.from_pretrained(self.lm_name)
        self.fc = nn.Linear(768, self.out_dim)

    def forward(self, x):
        x = x[0]
        out = self.lm(x)[-1]
        out = self.fc(out)
        return out
