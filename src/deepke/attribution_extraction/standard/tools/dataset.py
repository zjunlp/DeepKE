import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import load_pkl

def collate_fn(cfg):
    
    def collate_fn_intra(batch):
        """
    Arg : 
        batch () : 数据集
    Returna : 
        x (dict) : key为词，value为长度
        y (List) : 关系对应值的集合
    """
        batch.sort(key=lambda data: data['seq_len'], reverse=True)

        max_len = batch[0]['seq_len']

        def _padding(x, max_len):
            return x + [0] * (max_len - len(x))

        x, y = dict(), []
        word, word_len = [], []
        head_pos, tail_pos = [], []
        pcnn_mask = []
        for data in batch:
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(data['seq_len'])
            y.append(int(data['att2idx']))

            if cfg.model_name != 'lm':
                head_pos.append(_padding(data['entity_pos'], max_len))
                tail_pos.append(_padding(data['attribute_value_pos'], max_len))
                if cfg.model_name == 'cnn':
                    if cfg.use_pcnn:
                        pcnn_mask.append(_padding(data['entities_pos'], max_len))

        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        y = torch.tensor(y)

        if cfg.model_name != 'lm':
            x['entity_pos'] = torch.tensor(head_pos)
            x['attribute_value_pos'] = torch.tensor(tail_pos)
            if cfg.model_name == 'cnn' and cfg.use_pcnn:
                x['pcnn_mask'] = torch.tensor(pcnn_mask)
            if cfg.model_name == 'gcn':
                # 没找到合适的做 parsing tree 的工具，暂时随机初始化
                B, L = len(batch), max_len
                adj = torch.empty(B, L, L).random_(2)
                x['adj'] = adj
        return x, y

    return collate_fn_intra


class CustomDataset(Dataset):
    """
    默认使用 List 存储数据
    """
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)


