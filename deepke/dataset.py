import torch
from torch.utils.data import Dataset
from deepke.utils import load_pkl


class CustomLMDataset(Dataset):
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)


def collate_fn_lm(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    lens = [len(data[0]) for data in batch]
    max_len = max(lens)

    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    sent_arr = []
    y_arr = []
    for data in batch:
        sent, data_y = data
        sent_arr.append(_padding(sent, max_len))
        y_arr.append(data_y)
    return torch.tensor(sent_arr), torch.tensor(y_arr)


class CustomDataset(Dataset):
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)


def collate_fn(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    lens = [len(data[0]) for data in batch]
    max_len = max(lens)

    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    sent_arr = []
    head_pos_arr = []
    tail_pos_arr = []
    mask_arr = []
    y_arr = []
    for data in batch:
        sent, head_pos, tail_pos, mask, data_y = data
        sent_arr.append(_padding(sent, max_len))
        head_pos_arr.append(_padding(head_pos, max_len))
        tail_pos_arr.append(_padding(tail_pos, max_len))
        mask_arr.append(_padding(mask, max_len))
        y_arr.append(data_y)
    return torch.tensor(sent_arr), torch.tensor(head_pos_arr), torch.tensor(
        tail_pos_arr), torch.tensor(mask_arr), torch.tensor(y_arr)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    vocab_path = 'data/out/vocab.pkl'
    train_data_path = 'data/out/train.pkl'
    vocab = load_pkl(vocab_path)

    train_dataset = CustomDataset(train_data_path)
    dataloader = DataLoader(train_dataset,
                            batch_size=4,
                            shuffle=False,
                            collate_fn=collate_fn)
    for idx, (*x, y) in enumerate(dataloader):
        sent, head_pos, tail_pos, mask = x

        raw_sents = []
        for i in range(4):
            raw_sent = [vocab.idx2word[i] for i in sent[i].numpy()]
            raw_sents.append(''.join(raw_sent))
        print(raw_sents, head_pos, tail_pos, mask, y, sep='\n\n')
        break
