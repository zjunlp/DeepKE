import torch
from torch.utils.data import Dataset
from deepke.utils import load_pkl
from deepke.config import config


class CustomDataset(Dataset):
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)


def collate_fn(batch):
    batch.sort(key=lambda data: data['seq_len'], reverse=True)

    max_len = 0
    for data in batch:
        if data['seq_len'] > max_len:
            max_len = data['seq_len']

    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    if config.model_name == 'LM':
        x, y = [], []
        for data in batch:
            x.append(_padding(data['lm_idx'], max_len))
            y.append(data['target'])

        return torch.tensor(x), torch.tensor(y)

    else:
        sent, head_pos, tail_pos, mask_pos = [], [], [], []
        y = []
        for data in batch:
            sent.append(_padding(data['word2idx'], max_len))
            head_pos.append(_padding(data['head_pos'], max_len))
            tail_pos.append(_padding(data['tail_pos'], max_len))
            mask_pos.append(_padding(data['mask_pos'], max_len))
            y.append(data['target'])
        return torch.tensor(sent), torch.tensor(head_pos), torch.tensor(tail_pos), torch.tensor(
            mask_pos), torch.tensor(y)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    vocab_path = '../data/out/vocab.pkl'
    train_data_path = '../data/out/train.pkl'
    vocab = load_pkl(vocab_path)

    train_dataset = CustomDataset(train_data_path)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for idx, (*x, y) in enumerate(dataloader):
        print(x)
        print(y)
        break

