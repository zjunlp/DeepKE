import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Capsule(nn.Module):
    def __init__(self, config):
        super(Capsule, self).__init__()

        # self.xxx = config.xxx
