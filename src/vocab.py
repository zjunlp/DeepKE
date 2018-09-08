# coding=utf-8
import numpy as np
from config import Config

Flags = Config.config


class Vocab(object):
	"""docstring for Vocab"""
	def __init__(self,):
		self.id2token = {}
		self.token2id = {}

		self.ner2id = {}
		self.rel2id = {}

		self.token_cnt = {}

		self.embed_dim = None
		self.embeddings = None
		self.pad_token = '<blank>'
		self.unk_token = '<unk>'

		self.initial_tokens = [self.pad_token, self.unk_token, 'E1', 'E2']
		for token in self.initial_tokens:
			self.add(token)

		if filename is not None:
			self.load_from_file(config.data_path + 'vocab.txt')

		self.init_rel()
		self.init_ner()

	def get_id(self, token):
		try:
			return self.token2id[token]
		except KeyError:
			return self.token2id[self.unk_token]

	def get_token(self, id):
		try:
			return self.id2token[id]
		except KeyError:
			return self.unk_token

	def size(self):
		return len(self.id2token)

	def load_from_file(self, file_path):
		'''
		使用分段的file或者直接只是token的file，可能需要确定下
		yzq: file_path: vocab.txt
		'''

		for line in open(file_path, 'r'):
			token = line.rstrip('\n')[0]
			self.add(token)

	def add(self, token):
		if token in self.token2id.keys():
			return
		else:
			idx = len(self.id2token)
			self.token2id[token] = idx
			self.id2token[idx] = token

	def randomly_init_embeddings(self, embed_dim):
		self.embed_dim = embed_dim
		self.embeddings = np.random.rand(self.size(), embed_dim)
		for token in [self.pad_token, self.unk_token,'E1','E2']:
			self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

	def load_pretrained_embeddings(self, embedding_path):
		# em.txt
		self.embeddings = np.zeros([self.size(), self.embed_dim])
		with open(embedding_path, 'r') as fin:
			for line in fin:
				contents = line.strip().split()
				token = contents[0].decode('utf8')
				self.embeddings[self.get_id(token)] = list(map(float, contents[1:]))

	def init_rel(self):
		for (i, rel) in enumerate(Flags.relations):
			self.rel2id[rel] = i+1
		self.rel2id['0'] = 0

	def init_ner(self):
		for (i, ner) in enumerate(Flags['ner']):
			self.ner2id[ner] = i+1

	def get_token_ids(self, tokens, padding=True):
		if padding:
			sequences = [self.get_id(self.pad_token)] * config.MAX_LEN
		else:
			sequences = np.zeros(len(tokens), dtype=int)
		for index, token in enumerate(tokens):
			sequences[i] = self.get_id(token)
		return sequences

	def get_rel_id(self, rel):
		try:
			return self.rel2id[rel]
		except:
			return -1

	def get_ner_id(self, ner):
		try:
			return self.ner2id[ner]
		except:
			return 0

	def get_ner_ids(self, ners, padding=True):
		if padding:
			ner_sequences = np.zeros(config.MAX_LEN, dtype=int)
		else:
			ner_sequences = np.zeros(len(ners), dtype=int)
		for index,ner in enumerate(ners):
			ner_sequences[index] = self.get_ner_id(ner)
		return ner_sequences

