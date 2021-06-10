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
        """
        将一段文本按照制定拆分规则，拆分成一个词汇List
        Args :
            text (String) : 所需拆分文本
            never_split (List) : 不拆分的词，默认为空
        Rerurn : 
            output_tokens (List): 拆分后的结果 
        """
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
        """
        删除文本中无效字符以及空白字符
        Arg :
            text (String) : 所需删除的文本
        Return :
            "".join(output) (String) : 删除后的文本
        """
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
        """
        使用jieba分词
        Args :
            text (String) : 所需拆分文本
            never_split (List) : 不拆分的词
        Return :
            tokens (List) : 拆分完的结果
        """
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
        """
        在CJK字符周围添加空格
        Arg :
            text (String) : 所需拆分文本
        Return :
            "".join(output) (String) : 添加完后的文本
        """
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
        """
        在空白和一些标点符号（如逗号或句点）上拆分文本
        Arg :
            text (String) : 所需拆分文本
        Return :
            tokens (List) : 分词完的结果
        """
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
        """
        进行基本的空白字符清理和分割
        Arg :
            text (String) : 所需拆分文本
        Return :
            tokens (List) : 分词完的结果
        """
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _run_strip_accents(self, text):
        """
        从文本中去除重音符号
        Arg :
            text (String) : 所需拆分文本
        Return :
            "".join(output) (String) : 去除后的文本

        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """
        通过标点符号拆分文本
        Args :
            text (String) : 所需拆分文本
            never_split (List) : 不拆分的词，默认为空
        Return :
            ["".join(x) for x in output] (List) : 拆分完的结果
        """
        
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
        """
        判断字符是否为控制字符
        Arg :
            char : 字符
        Return :
            bool : 判断结果
        """
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
        """
        判断字符是否为空白字符
        Arg :
            char : 字符
        Return :
            bool : 判断结果
        """
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
        """
        
        判断字符是否为中文字符
        Arg :
            cp (char): 字符
        Return :
            bool : 判断结果
        
        """
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
        """
        判断字符是否为标点字符
        Arg :
            char : 字符
        Return :
            bool : 判断结果
        """
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
