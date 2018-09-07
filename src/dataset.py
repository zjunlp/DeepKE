# coding=UTF-8

import tensorflow as tf
import codecs
import re
import numpy as np
import pickle
import pandas as pd
from vocab import Vocab
from config import Config
from utility import Tools
config = Config.config
v = Vocab()
tools = Tools()


class Dataset(object):
	def __init__(self, file_path):
		self.data = self.__load_dataset(self.file_path)

	def __load_dataset(self, data_path,):
		'''
		假设读入文件格式为csv: [rel,]sent,e1,e2,pos1,pos2,[ner]
		'''
		content = pd.read_csv(data_path, sep='\t')
		data = []
		need_ner = False
		if 'ner' in config.model:
			need_ner = True
		for index, row in content.iterrows():
			rel = row.relations
			sent = row.sentences
			ent1pos = row.pos1
			ent2pos = row.pos2
			b,d = [ent1pos,ent2pos] if ent1pos < ent2pos else [ent2pos,ent1pos]
			mask = self.__mask(len(sent),ent1pos,ent2pos)

			rel2id = v.get_rel_id(rel)
			sent2id = v.get_token_ids(sent, padding=True)
			pos1 = []
			pos2 = []
			for idx, word in enumerate(sent):
				position1 = idx - ent1pos
				position2 = idx - ent2pos
				pos1.append(position1)
				pos2.append(position2)
			if need_ner:
				try:
					ner = row.ner
				except:
					ner = tools.ner_one(sent, ner_way=config.ner_way)
				ner2id = v.get_ner_ids(ner, padding=True)
			else:
				ner2id = np.zeros(config.max_len, dtype=int)
			d = [rel2id, sent2id, pos1, pos2, ner2id, mask]
			data.append(d)
		return data

	def __mask(self, length,b,d):
		L = config.MAX_LEN
		mask = np.zeros((3, L), dtype=np.float32)

		# piecewise cnn: 0...b-1; b ... d-1; d ... L
		if b > 0:
			mask[0, :b] = 1
		if b < d:
			mask[1, b:d] = 1
		if d < length:
			mask[2, d:length] = 1
		return mask

	def one_mini_batch(self,batch_size, shuffle=True):
		data = self.data
		data = np.array(data)
		data_size = len(data)

		batches_per_epoch = data_size // batch_size

		if shuffle:
			indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[indices]
		else:
			shuffled_data = data

		for batch_num in range(batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index: end_index]

	def get_data(self):
		return self.data


