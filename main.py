import os
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from deepke.config import config
from deepke import model
from deepke.utils import make_seed, load_pkl
from deepke.trainer import train, validate
from deepke.preprocess import process
from deepke.dataset import CustomDataset, collate_fn

warnings.filterwarnings("ignore")

__Models__ = {
    "CNN": model.CNN,
    "RNN": model.BiLSTM,
    "GCN": model.GCN,
    "Transformer": model.Transformer,
    "Capsule": model.Capsule,
    "LM": model.LM,
}

parser = argparse.ArgumentParser(description='choose your model')
parser.add_argument('--model', type=str, help='model name')
args = parser.parse_args()
model_name = args.model if args.model else config.model_name

make_seed(config.training.seed)

if config.training.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda', config.training.gpu_id)
else:
    device = torch.device('cpu')

# if not os.path.exists(config.out_path):
process(config.data_path, config.out_path)

train_data_path = os.path.join(config.out_path, 'train.pkl')
test_data_path = os.path.join(config.out_path, 'test.pkl')

if model_name == 'LM':
    vocab_size = None
else:
    vocab_path = os.path.join(config.out_path, 'vocab.pkl')
    vocab = load_pkl(vocab_path)
    vocab_size = len(vocab.word2idx)

train_dataset = CustomDataset(train_data_path)
train_dataloader = DataLoader(train_dataset,
                              batch_size=config.training.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
test_dataset = CustomDataset(test_data_path)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.training.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

model = __Models__[model_name](vocab_size, config)
model.to(device)
# print(model)

optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'max',
    factor=config.training.decay_rate,
    patience=config.training.decay_patience)
criterion = nn.CrossEntropyLoss()

best_macro_f1, best_macro_epoch = 0, 1
best_micro_f1, best_micro_epoch = 0, 1
best_macro_model, best_micro_model = '', ''
print('=' * 10, ' Start training ', '=' * 10)

for epoch in range(1, config.training.epoch + 1):
    train(epoch, device, train_dataloader, model, optimizer, criterion, config)
    macro_f1, micro_f1 = validate(test_dataloader, model, device, config)
    model_name = model.save(epoch=epoch)
    scheduler.step(macro_f1)

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_macro_epoch = epoch
        best_macro_model = model_name
    if micro_f1 > best_micro_f1:
        best_micro_f1 = micro_f1
        best_micro_epoch = epoch
        best_micro_model = model_name

print('=' * 10, ' End training ', '=' * 10)
print(f'best macro f1: {best_macro_f1:.4f},',
      f'in epoch: {best_macro_epoch}, saved in: {best_macro_model}')
print(f'best micro f1: {best_micro_f1:.4f},',
      f'in epoch: {best_micro_epoch}, saved in: {best_micro_model}')
