# coding=utf-8
import logging
import sys
from NER.ner import zjuner
from config import Config
import re
import jieba

config = Config.config

class DottableDict(dict):
	# 这样dict能通过点访问如 config.max_len
	def __init__(self, *args, **kwargs):
		dict.__init__(self, *args, **kwargs)
		self.__dict__ = self

	def allowDotting(self, state=True):
		if state:
			self.__dict__ = self
		else:
			self.__dict__ = dict()


class Logger(object):
	def __init__(self,name, log_file=None):
		self.logger = self.__get_logger(name,log_path=None)

	def __get_logger(self,name,log_path=None):
		logger = logging.getLogger(name)
		formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s : %(message)s')
		logger.setLevel(logging.DEBUG)
		if log_path is not None:
			handler = logging.FileHandler(log_path)
		else:
			handler = logging.StreamHandler(sys.stdout)
		handler.setLevel(logging.INFO)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		return logger


class Tools(object):
	def clean(self,sentences, is_training=True):
		'''

		1. 替换句子中的一些字符，替换NUM，
		2. min_character <限制句子长度 <max_character ，对>config.max_character进行分句，注意
		:param sentences: sent_list
		:param is_training: 如果是对原始数据的清理，清理完之后直接加个分词来训练embedding并且得到vocab.txt
		:return: new_sent_list (len(new_sent_list) might not equal to sent_list)
		'''

		ret = []
		for sent in sentences:
			while(len(sent) > config.min_charater):
				if(len(sent) > config.max_charater):
					s = sent[0:config.max_charater]
					sent = sent.replace(s,'',1)
				else:
					s = sent
				s = re.sub(r'[0-9]+', 'NUM', sent)
				s = sent.replace("\t", " ")
				s = sent.replace("\"", " ")
				s = sent.replace('“', '')
				s = sent.replace('”', '')
				s = sent.replace(" ", "")
				s = sent.replace("|", "")
				s = sent.replace(".", "")
				ret.append(sent)
		return ret

	def __ner_self(self,sentence):
		seg_sent = ''
		# ...
		return seg_sent

	def __ner_fudan(self,sentence):
		pass

	def __ner_zju(self,sent):
		tag = ner(sent)
		return tag

	def ner_one(self,sentence,ner_way):
		if ner_way == 'self':
			return self.__ner_self(sentence)
		if ner_way == 'fudan':
			return self.__ner_fudan(sentence)
		if ner_way == 'zju':
			return self.__ner_zju(sentence)

	def ner_all(self,sentences,ner_way):
		ret = []
		for sentence in sentences:
			ret.append(self.ner_one(sentence))
		return ret

	def __segment_jieba(self,sentence):
		if config.extra_dict_path is None:
			return ' '.join()
		else:
			jieba.load_userdict(config.extra_dict_path)
		sent = jieba.cut(sentence)
		sent = ' '.join(sent)
		return sent

	def __segment_character(self,sentence):
		# character_level (注意英文,E1,E2,关键字)
		'''yzq：只考虑了E1,E2'''
		ret = ''
		for w in sentence:
			ret += w
			ret = ' '
		ret = re.sub(r'E\s1','E1',ret)
		ret = re.sub(r'E\s2','E2',ret)
		ret = re.sub(r'N\sU\sM', 'NUM', ret)
		return ret

	def segment_one(self,sentence,seg_way):
		# 分段通常意味着position的更新，
		if seg_way=='character':
			return self.__segment_character(sentence)
		if seg_way=='jieba':
			return self.__segment_jieba(sentence)

	def segment_all(self,sentences,seg_way):
		ret = []
		for sentence in sentences:
			ret.append(sentence,seg_way)
		return ret

	def relocate_postion(self,seg_sent,e1,e2,old_pos1,old_pos2):
		'''
		注意这里的seg_sent是已经用'E1'/'E2'替换掉e1/e2的分词/分句后的，以' '间隔的句子
		看是不是需要放在sample的 __replace_entity中，或者把__replace_entity移到此处
		:param seg_sent:
		:param e1:
		:param e2:
		:param old_pos1:
		:param old_pos2:
		:return:
		'''
		sent = re.sub(r'\s', '', seg_sent)
		p1 = re.search('E1', sent)
		p2 = re.search('E2', sent)
		return p1,p2




