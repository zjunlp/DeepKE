import torch
from torchtext import data
from models import BiLSTM
from dataset import build_vocab
from train import train
from config import config, set_seed

set_seed(config.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, valid_data, sent_vocab, label_vocab = build_vocab(config)
vocab_size = len(sent_vocab)

train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_sizes=(config.batch_size, config.batch_size),
    sort_key=lambda x: len(x.sentence),
    sort_within_batch=True,
    device=device)

model = BiLSTM(vocab_size, config.embedding_size, config.hidden_size,
               config.output_size)

f1_micro_best, f1_macro_best = 0, 0
for epoch in range(1, config.epochs):
    f1_macro, f1_micro = train(epoch, train_iter, valid_iter, model, config,
                               device)
    if f1_macro > f1_macro_best:
        f1_macro_best = f1_macro
    if f1_micro > f1_micro_best:
        f1_micro_best = f1_micro

print('\n\n', '=' * 30)
print('during training, best f1 score: [macro / {:.2f}] [micro / {:.2f}]'.format(
    f1_macro_best, f1_micro_best))
