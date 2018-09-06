# coding=utf-8
from gensim.models.word2vec import Word2Vec
from config import Config
import gensim
import os
from utility import Logger
logger = Logger(name='Sample.py')
logging = logger.logger

Flags = Config.config


def bin2txt(bin_file_path):
	'''
	如果是已经训练的模型是*.bin，转换下
	:param path: ../data/embedding/words.bin
	:return:
	'''
	from gensim.models import KeyedVectors
	word_vectors = KeyedVectors.load_word2vec_format('words.bin', binary=True)  # C binary format
	word_vectors.wv.save_word2vec_format(save_path + 'em.txt', fvocab=save_path + 'vocab.txt', binary=False)


def w2v(save_path, segment_sentences):
	'''
	可以用已经训练好的或者用segment_sentences生成
	'''
	logger.info('\n train embedding')
	# sentences = MySentences(Flags.corpus_path) # 建议名字为 'original_text_seg.txt'
	model = Word2Vec(segment_sentences, size=Flags.embedding_dim, window=5, min_count=5, workers=4)
	model.wv.save_word2vec_format(save_path + 'em.txt', fvocab=save_path + 'vocab.txt', binary=False)


if __name__ == '__main__':
	with open(Flags.data_path + 'seg_original_data.txt') as f:
		sentences = [line.strip() for line in f]
	w2v(Flags.embedding_path, sentences)


