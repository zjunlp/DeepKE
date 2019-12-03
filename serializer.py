import re
import unicodedata
import jieba
import logging
from typing import List

logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.INFO)


class Serializer():
    def __init__(self, never_split: List = None, do_lower_case=True, do_chinese_split=False):
        self.never_split = never_split if never_split is not None else []
        self.do_lower_case = do_lower_case
        self.do_chinese_split = do_chinese_split

    def serialize(self, text, never_split: List = None):
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)

        if self.do_chinese_split:
            output_tokens = self._use_jieba_cut(text, never_split)
            return output_tokens

        text = self._tokenize_chinese_chars(text)
        orig_tokens = self._orig_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split=never_split))

        output_tokens = self._whitespace_tokenize(" ".join(split_tokens))

        return output_tokens

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self.is_control(char):
                continue
            if self.is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _use_jieba_cut(self, text, never_split):
        for word in never_split:
            jieba.suggest_freq(word, True)
        tokens = jieba.lcut(text)
        if self.do_lower_case:
            tokens = [i.lower() for i in tokens]
        try:
            while True:
                tokens.remove(' ')
        except:
            return tokens

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self.is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _orig_tokenize(self, text):
        """Splits text on whitespace and some punctuations like comma or period"""
        text = text.strip()
        if not text:
            return []
        # 常见的断句标点
        punc = """,.?!;: 、｜，。？！；：《》「」【】/<>|\“ ”‘ ’"""
        punc_re = '|'.join(re.escape(x) for x in punc)
        tokens = re.sub(punc_re, lambda x: ' ' + x.group() + ' ', text)
        tokens = tokens.split()
        return tokens

    def _whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self.is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    @staticmethod
    def is_control(char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    @staticmethod
    def is_whitespace(char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    @staticmethod
    def is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    @staticmethod
    def is_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96)
                or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False


if __name__ == '__main__':
    text1 = "\t\n你   好呀， I\'m his pupp\'peer,\n\t"
    text2 = '你孩子的爱情叫 Stam\'s 的打到天啊呢哦'

    serializer = Serializer(do_chinese_split=False)
    print(serializer.serialize(text1))
    print(serializer.serialize(text2))

    text3 = "good\'s  head  pupp\'er, "
    # print: ["good's", 'pupp', "'", 'er', ',']
    # true:  ["good's", "pupp'er", ","]
    print(serializer.serialize(text3, never_split=["pupp\'er"]))
