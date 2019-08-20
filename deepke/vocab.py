from typing import List

init_tokens = ['PAD', 'UNK']


class Vocab(object):
    def __init__(self, name: str, init_tokens: List[str] = init_tokens):
        self.name = name
        self.init_tokens = init_tokens
        self.trimed = False
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {}
        self.count = 0
        self.add_init_tokens()

    def add_init_tokens(self):
        for token in self.init_tokens:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.count
            self.word2count[word] = 1
            self.idx2word[self.count] = word
            self.count += 1
        else:
            self.word2count[word] += 1

    def add_sent(self, sent: str):
        for word in sent:
            self.add_word(word)

    def trim(self, min_freq=2, verbose: bool = True):
        '''
        当 word 词频低于 min_freq 时，从词库中删除
        :param min_freq: 最低词频
        '''
        if self.trimed:
            return
        self.trimed = True

        keep_words = []
        new_words = []
        for k, v in self.word2count.items():
            if v >= min_freq:
                keep_words.append(k)
                new_words.extend([k] * v)
        if verbose:
            print('after trim, keep words [{} / {}] = {:.2f}%'.format(
                len(keep_words + self.init_tokens), len(self.word2idx),
                len(keep_words + self.init_tokens) / len(self.word2idx) * 100))

        # Reinitialize dictionaries
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {}
        self.count = 0
        self.add_init_tokens()
        for word in new_words:
            self.add_word(word)


if __name__ == '__main__':
    from nltk import word_tokenize
    vocab = Vocab('test')
    sent = ' 我是中国人，我爱中国。'
    # english
    # sent = "I'm chinese, I love China."
    # words = word_tokenize(sent)
    print(sent, '\n')
    vocab.add_sent(sent)
    print(vocab.word2idx)
    print(vocab.word2count)
    vocab.trim(2)
    print(vocab.word2idx)
