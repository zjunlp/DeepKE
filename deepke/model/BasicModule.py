import torch
import torch.nn as nn
import time
from deepke.utils import ensure_dir


class BasicModule(nn.Module):
    '''
    封装nn.Module, 提供 save 和 load 方法
    '''
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        '''
        加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, epoch=0, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        prefix = 'checkpoints/'
        ensure_dir(prefix)
        if name is None:
            name = prefix + self.model_name + '_' + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + name + '_' + self.model_name + '_' + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name