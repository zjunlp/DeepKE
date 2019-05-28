import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from vocab import Vocab
from models import BiLSTM_ATT
from dataset import CustomDataset
from metric import PRMetric
from utils import set_seed, snapshot
from config import config

# logging config
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# reproducibility
set_seed(config.seed)

# save path
save_path = config.save_path

# vocab
vocab = Vocab(min_count=config.min_count,
              data_path=config.train_data_path,
              vocab_path=config.vocab_path)
print('vocab is building in path: {}.'.format(config.vocab_path))

# dataset
train_dataset = CustomDataset(fp=config.train_data_path, min_count=2)
test_dataset = CustomDataset(fp=config.test_data_path, min_count=2)
train_data = TensorDataset(train_dataset.train_x, train_dataset.train_l,
                           train_dataset.train_y)
test_data = TensorDataset(test_dataset.train_x, test_dataset.train_l,
                          test_dataset.train_y)
# data loader
batch_size = config.batch_size
train_loader = DataLoader(train_data,
                          batch_size,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True)
test_loader = DataLoader(test_data,
                         batch_size,
                         shuffle=False,
                         num_workers=0,
                         drop_last=True)

# model
vocab_size = len(vocab.word2id) + 2
embedding_size = config.embedding_size
hidden_size = config.hidden_size
output_size = len(train_dataset.rel2id)
net = BiLSTM_ATT(vocab_size,
                 embedding_size,
                 hidden_size,
                 output_size=output_size)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(net.parameters(), lr=config.lr)

# device
device = torch.device(
    'cuda: %s' %
    config.gpu if torch.cuda.is_available() and config.gpu >= 0 else 'cpu')
print('device:', device)
net = net.to(device)

# f1, p, r
train_metrics = PRMetric(num_class=output_size)

# train
for epoch in range(1, config.epochs + 1):
    net.train()
    for batch_i, (x, x_len,
                  y) in tqdm(enumerate(train_loader),
                             total=len(train_loader) // batch_size + 1):
        x, x_len, y = x.to(device), x_len.to(device), y.to(device)

        y_pred = net(x, x_len)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Loss: {:.4f}".format(loss.item()))

    # 在test数据集上计算 p, r, f1
    net.eval()
    train_metrics.reset()
    for batch_i, (x, x_len, y) in tqdm(enumerate(test_loader),
                                       total=len(test_data) // batch_size + 1):
        x, x_len = x.to(device), x_len.to(device)
        with torch.no_grad():
            y_pred = net(x, x_len)
        train_metrics.update((y_pred, y))
    p, r = train_metrics.compute()
    p = np.average(p)
    r = np.average(r)
    if p == 0 and r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    logging.info('f1: {0:.4f}, p: {1:.4f}, r: {2:.4f}'.format(f1, p, r))

    # 保存模型
    snapshot(net, epoch, save_path)
