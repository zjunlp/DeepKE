import os
import time
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    '''
    封装nn.Module, 提供 save 和 load 方法
    '''
    def __init__(self):
        super(BasicModule, self).__init__()


    def load(self, path, device):
        '''
        加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path, map_location=device))


    def save(self, epoch=0, cfg=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S')
        prefix = os.path.join(cfg.cwd, 'checkpoints',time_prefix)
        os.makedirs(prefix, exist_ok=True)
        name = os.path.join(prefix, cfg.model_name + '_' + f'epoch{epoch}' + '.pth')

        torch.save(self.state_dict(), name)
        return name



