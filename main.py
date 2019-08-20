import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from deepke.config import config
from deepke import model
from deepke.utils import make_seed, load_pkl
from deepke.trainer import train, validate
from deepke.process import process
from deepke.dataset import CustomDataset, CustomLMDataset, collate_fn, collate_fn_lm

__Models__ = {
    "CNN": model.CNN,
    "BiLSTM": model.BiLSTM,
    "Transformer": model.Transformer,
    "Capsule": model.Capsule,
    "Bert": model.Bert,
}

parser = argparse.ArgumentParser(description='choose your model')
parser.add_argument('--model_name', type=str, default='CNN', help='model name')
args = parser.parse_args()
model_name = args.model_name if args.model_name else config.model_name

make_seed(config.seed)

if config.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda', config.gpu_id)
else:
    device = torch.device('cpu')

if not os.path.exists(config.out_path):
    process(config.data_path, config.out_path, file_type='csv')

if config.model_name == 'Bert':
    vocab_path = os.path.join(config.out_path, 'bert_vocab.txt')
    train_data_path = os.path.join(config.out_path, 'train_lm.pkl')
    test_data_path = os.path.join(config.out_path, 'test_lm.pkl')
else:
    vocab_path = os.path.join(config.out_path, 'vocab.pkl')
    train_data_path = os.path.join(config.out_path, 'train.pkl')
    test_data_path = os.path.join(config.out_path, 'test.pkl')

vocab = load_pkl(vocab_path)
vocab_size = len(vocab.word2idx)

if config.model_name == 'Bert':
    train_dataset = CustomLMDataset(train_data_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn_lm)
    test_dataset = CustomLMDataset(test_data_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn_lm,
    )
else:
    train_dataset = CustomDataset(train_data_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataset = CustomDataset(test_data_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

model = __Models__[model_name](vocab_size, config)
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', factor=config.decay_rate, patience=config.decay_patience)
criterion = nn.CrossEntropyLoss()

best_macro_f1, best_macro_epoch = 0, 1
best_micro_f1, best_micro_epoch = 0, 1
best_macro_model, best_micro_model = '', ''
print('=' * 10, ' Start training ', '=' * 10)

for epoch in range(1, config.epoch + 1):
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
