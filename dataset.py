import re
import jieba
from torchtext import data


def build_vocab(config):
    tokenizer = lambda s: re.split('', s)

    TEXT = data.Field(sequential=True,
                      lower=True,
                      include_lengths=True,
                      batch_first=True,
                      tokenize=tokenizer)
    LABEL = data.Field(sequential=False, use_vocab=True, unk_token=None)

    train_data, valid_data = data.TabularDataset.splits(path=config.data_path,
                                                        train='train.csv',
                                                        validation='valid.csv',
                                                        format='csv',
                                                        skip_header=True,
                                                        fields=[
                                                            ('sentence', TEXT),
                                                            ('entity1', None),
                                                            ('entity2', None),
                                                            ('offset1', None),
                                                            ('offset2', None),
                                                            ('relation', LABEL)
                                                        ])

    TEXT.build_vocab(train_data, min_freq=2)
    LABEL.build_vocab(train_data)

    sent_vocab = TEXT.vocab
    label_vocab = LABEL.vocab
    return train_data, valid_data, sent_vocab, label_vocab


if __name__ == '__main__':
    from config import config
    train_data, valid_data, sent_vocab, label_vocab = build_vocab(config)

    print(f'vocab length: {len(sent_vocab)}')
    print(label_vocab.stoi)
    print(sent_vocab.stoi)  # this store word2idx
    # print(sent_vocab.freqs)   # this store word2count
    # idx2word = {value: key for key, value in sent_vocab.stoi.items()}
    # print(idx2word)           # this store idx2word
