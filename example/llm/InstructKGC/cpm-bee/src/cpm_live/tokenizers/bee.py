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

import pkg_resources
import io
from typing import IO, Dict, List, Optional, Tuple


def load_vocab(fp: IO[bytes]) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary."""
    vocab: Dict[str, int] = {}

    reader = io.TextIOWrapper(fp, encoding="utf-8")
    for token in reader.readlines():
        if token[-1] == "\n":
            token = token[:-1]
        if len(token) == 0:
            continue
        vocab[token] = len(vocab)
    return vocab


class Token(object):
    def __init__(self, token: str, start: int, is_unk: bool, is_special: bool):
        self.token = token
        self.start = start
        self.is_unk = is_unk
        self.is_special = is_special

    def __str__(self):
        return "Token(token={}, start={}, is_unk={}, is_special={})".format(
            self.token, self.start, self.is_unk, self.is_special
        )

    def __repr__(self):
        return self.__str__()


class CPMBeeTokenizer(object):
    def __init__(
        self,
    ):
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.line_token = "\n"
        self.space_token = " "

        self.encoder = load_vocab(pkg_resources.resource_stream("cpm_live", "vocabs/bee.txt"))
        self.encoder[self.line_token] = self.encoder["</n>"]
        self.encoder[self.space_token] = self.encoder["</_>"]
        del self.encoder["</n>"]
        del self.encoder["</_>"]

        self.decoder = {v: k for k, v in self.encoder.items()}
        self._special_tokens = {
            k: v for k, v in self.encoder.items() if k.startswith("<") and k.endswith(">")
        }

        self._max_word_len = max([len(x) for x in self.encoder.keys()])

    def get_piece(self, text: str) -> str:
        text = text[: self._max_word_len]
        len_text = len(text)
        for i in range(len(text)):
            sub = text[: len_text - i]
            if (sub in self.encoder) and (sub not in self._special_tokens):
                return sub
        return text[0]

    @property
    def vocab_size(self):
        return len(self.encoder)

    @property
    def eos_id(self):
        return self.encoder[self.eos_token]

    @property
    def bos_id(self):
        return self.encoder[self.bos_token]

    @property
    def unk_id(self):
        return self.encoder[self.unk_token]

    @property
    def mask_id(self):
        return self.encoder[self.mask_token]

    def __len__(self):
        return len(self.encoder)

    def tokenize(self, text: str) -> List[Token]:
        output_tokens: List[Token] = []

        sentence_split = [""]
        is_escape = False
        is_special_token = False
        for i, c in enumerate(text):
            if is_special_token:
                if c == "<":
                    raise ValueError("Invalid special token at pos {}".format(i))
                elif c == ">":
                    # end of special token
                    sentence_split[-1] += c
                    is_special_token = False
                    sentence_split.append("")
                else:
                    sentence_split[-1] += c
            else:
                if c == "<":
                    if is_escape:
                        # case: <<
                        sentence_split[-1] += c
                        is_escape = False
                    else:
                        # case: x<
                        is_escape = True
                else:
                    if is_escape:
                        # case <x
                        is_special_token = True
                        is_escape = False
                        sentence_split.append("<" + c)
                    else:
                        # case xx
                        sentence_split[-1] += c
        if is_escape or is_special_token:
            raise ValueError("Unexpected end of text `{}`".format(text))

        part_pos = 0
        for i, part in enumerate(sentence_split):
            if (i & 1) == 1:
                # special token
                output_tokens.append(Token(part, part_pos, False, True))
            else:
                part_st = 0
                last_unk = None
                while part_st < len(part):
                    piece = self.get_piece(part[part_st:])
                    if piece not in self.encoder:
                        if last_unk is None:
                            last_unk = piece
                        else:
                            last_unk += piece
                    else:
                        if last_unk is None:
                            output_tokens.append(Token(piece, part_st + part_pos, False, False))
                        else:
                            output_tokens.append(
                                Token(last_unk, part_st + part_pos - len(last_unk), True, False)
                            )
                            output_tokens.append(Token(piece, part_st + part_pos, False, False))
                            last_unk = None
                    part_st += len(piece)
                if last_unk is not None:
                    # part end with UNK
                    output_tokens.append(
                        Token(last_unk, part_st + part_pos - len(last_unk), True, False)
                    )
            part_pos += len(part)
        return output_tokens

    @staticmethod
    def escape(text: str) -> str:
        return text.replace("<", "<<")

    @staticmethod
    def unescape(text: str) -> str:
        return text.replace("<<", "<")

    def encode(
        self, text: str, past_table: Dict[int, str] = {}
    ) -> Tuple[List[int], Dict[int, str]]:
        ext_table_rev: Dict[str, int] = {}
        ext_table: Dict[int, str] = {}
        for idx, val in past_table.items():
            ext_table[idx] = val
            ext_table_rev[val] = idx
        ret = []
        for x in self.tokenize(text):
            if x.is_unk or (x.is_special and (x.token not in self.encoder)):
                if x.token not in ext_table_rev:
                    ext_table_rev[x.token] = len(ext_table_rev) + self.vocab_size
                    ext_table[ext_table_rev[x.token]] = x.token
                ret.append(ext_table_rev[x.token])
            elif x.token in self.encoder:
                ret.append(self.encoder[x.token])
            else:
                raise ValueError("Unknown token `{}` at pos {}".format(x.token, x.start))

        return ret, ext_table

    def decode(self, tokens: List[int], ext_table: Optional[Dict[int, str]] = None):
        """Decode ids into a string."""
        if ext_table is None:
            ext_table = {}
        ret = []
        for token in tokens:
            if token in ext_table:
                ret.append(ext_table[token])
            else:
                if token >= 0:
                    w = self.decoder[token]
                    if w in self._special_tokens:
                        ret.append(w)
                    else:
                        ret.append(self.escape(w))
        return "".join(ret)
