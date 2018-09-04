# coding=UTF-8
 
import tensorflow as tf
from config import Config
config =


class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self, train_files=[], predict_files=[]):
		super(Dataset, self).__init__()
		self.train_file = []
		self.predict_file = []
		# 调用__load_dataset加载到
		self.train_data,self.test_data,self.predict_data = [], [], []

	def clean(self, data_path):
		'''
		清理一些符号等，替换NUM；
		'''
		pass

	def gen_kg():
		'''
		DS数据来源，
		无论是哪种渠道生成，最后都是self.kg_data,
		无论是为ner还是re任务作标注，self.kg_data格式都是e1,e2,relations(所有的关系)
		'''
		pass

	def _load_dataset(self, data_path, train=False):
		'''
		
		'''
		pass


	def _segment(self,train=False,character_level = False):
		'''
		分词/字,（替换E1,E2）
		'''
		pass

	def _locate_entity_position(self,train=False):
		'''
		两个实体位置，加入pos1, pos2
		'''
		pass

	def _ner(self,train=False,ner_tool='zju'):
		'''
		
		'''
		pass

	def normalize_data(self,train=False,ner_tool='zju'):
		'''
		要用上面三个函数
		'''
		pass


	def _one_mini_batch(self, data, indices, pad_id,type='re'):
		'''
		也可以直接在gen_batches进行
		'''
		if type== 're':
			rels,sents,e1s,e2s,pos1s,pos2s,ners = [] *7
			for sidx,sample in enumerate(samples):
					rels.append(sample['rel']) 
					# ...
			batch_data ={'rel':rels,
				 'sent':sents,
				 'e1':e1s,
				 'e2':e2s,
				 'pos1':pos1s,
				 'pos2':pos2s,
				 'ner':ners,}
		elif type == 'ner':
			pass
		elif type == 'joint':
			pass
		return _dynamic_padding(batch_data)

	def _dynamic_padding(self, data, pad_id):
		pass

	def _mask(data):
		'''
		给原始数据中添加 mask
		:return: 
		'''
		return data

	def convert_to_ids(self, vocab):
		'''
		需要转换 rel,sentence,e1,e2,pos1,pos2,ner
		可能用上面三个
		'''
		pass

	def gen_batches(self, set_name, batch_size, pad_id, shuffle=True,train=False):
		'''
		输入data，之后用json格式，输入的data里包括, rel,sentences(分词的/或者分成单个word的),e1,e2,pos1,pos2
		会用normalize_data，选用_one_mini_batch(data)
		'''
		pass

		