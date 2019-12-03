import pytest
from serializer import Serializer
from vocab import Vocab


def test_vocab():
    vocab = Vocab('test')
    sent = ' 我是中国人，我爱中国。 I\'m Chinese, I love China'

    serializer = Serializer(do_lower_case=True)
    tokens = serializer.serialize(sent)
    assert tokens == [
        '我', '是', '中', '国', '人', '，', '我', '爱', '中', '国', '。', 'i', "'", 'm', 'chinese', ',', 'i', 'love', 'china'
    ]

    vocab.add_words(tokens)
    unk_str = '[UNK]'
    unk_idx = vocab.word2idx[unk_str]

    assert vocab.count == 22
    assert len(vocab.word2idx) == len(vocab.idx2word) == len(vocab.word2idx) == 22

    vocab.trim(2, verbose=False)

    assert vocab.count == 11
    assert len(vocab.word2idx) == len(vocab.idx2word) == len(vocab.word2idx) == 11

    token2idx = [vocab.word2idx.get(i, unk_idx) for i in tokens]
    assert len(tokens) == len(token2idx)
    assert token2idx == [7, 1, 8, 9, 1, 1, 7, 1, 8, 9, 1, 10, 1, 1, 1, 1, 10, 1, 1]

    idx2tokens = [vocab.idx2word.get(i, unk_str) for i in token2idx]
    assert len(idx2tokens) == len(token2idx)
    assert ' '.join(idx2tokens) == '我 [UNK] 中 国 [UNK] [UNK] 我 [UNK] 中 国 [UNK] i [UNK] [UNK] [UNK] [UNK] i [UNK] [UNK]'


if __name__ == '__main__':
    pytest.main()
