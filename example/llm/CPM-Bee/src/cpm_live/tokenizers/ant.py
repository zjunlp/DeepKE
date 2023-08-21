# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jieba
import pkg_resources
import io
from typing import IO


def load_vocab(fp: IO[bytes]):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}

    reader = io.TextIOWrapper(fp, encoding="utf-8")
    for token in reader.readlines():
        token = token.strip()
        if len(token) == 0:
            continue
        vocab[token] = len(vocab)
    return vocab


class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
            else:
                sub_tokens.append(cur_substr)
                start = end

        return sub_tokens


class CPMAntTokenizer(object):
    def __init__(
        self,
        bod_token="<d>",
        eod_token="</d>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        line_token="</n>",
        space_token="</_>",
    ):

        self.bod_token = bod_token
        self.eod_token = eod_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.line_token = line_token
        self.space_token = space_token

        self.encoder = load_vocab(pkg_resources.resource_stream("cpm_live", "vocabs/ant.txt"))
        self.encoder[" "] = self.encoder[space_token]
        self.encoder["\n"] = self.encoder[line_token]

        del self.encoder[self.space_token]
        del self.encoder[self.line_token]

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder, unk_token=self.unk_token)

    @property
    def vocab_size(self):
        return len(self.encoder)

    @property
    def bod_id(self):
        return self.encoder[self.bod_token]

    @property
    def eod_id(self):
        return self.encoder[self.eod_token]

    @property
    def eos_id(self):
        return self.encoder[self.eos_token]

    @property
    def bos_id(self):
        return self.encoder[self.bos_token]

    @property
    def pad_id(self):
        return self.encoder[self.pad_token]

    @property
    def unk_id(self):
        return self.encoder[self.unk_token]

    @property
    def newline_id(self):
        return self.encoder["\n"]

    def __len__(self):
        return len(self.encoder)

    def tokenize(self, text):
        """Tokenize a string."""
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens

    def encode(self, text):
        """Encode a string into ids."""
        return [self.encoder[x] for x in self.tokenize(text)]

    def decode(self, tokens):
        """Decode ids into a string."""
        tokens = [i for i in tokens if i >= 0]
        text = "".join([self.decoder[x] for x in tokens])
        return text

    def check(self, token):
        return token in self.encoder

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder.get(x, self.encoder[self.unk_token]) for x in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.decoder[x] if x >= 0 else self.unk_token for x in ids]
