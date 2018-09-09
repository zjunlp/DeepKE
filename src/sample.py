# coding=UTF-8
from config import Config
config = Config.config
from utility import Logger
logger = Logger(name='Sample.py')
logging = logger.logger
from multiprocessing import Process
import numpy as np
import re
import os
import sys
import pandas as pd

if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")


class Sample():
	def __init__(self,data_src='file',src_path=None):
		logging.info('远程监督的数据路径%s'%src_path)
		self.get_ds_data(data_src,src_path)

	def get_ds_data(self,data_src='file',src_path=None):
		'''
		如果是从文件中读取，文件里存的是 [relation		e1	e2	]
		:param data_src: ['file','db'] 数据来源
		'''
		logging.info('加载远程监督数据')
		ds_data = [] # [relation, e1,e2]
		if data_src == 'file':
			with open(src_path) as f:
				lines = f.readlines()
			lines = list(set(lines))
			for line in lines:
				parts = line.strip().split('\t')
				if len(parts) != 3:
					continue
				ds_data.append(parts)
		elif data_src == 'db':
			logging.warning('目前没有写从数据库读取远程监督数据来源部分')
			exit()
		else:
			logging.error("远程监督的数据来源['file','db']")
			exit()

		# 注意如果是公司和公司的关系的，e1_pool和e2_pool可能是相同的，不需要两个pool，
		# 考虑到DS不全，不建议从较全的e1或者e2中读取相应的pool,可以能减少错误样本的标记
		all_relations = list(set([each[0] for each in ds_data]))
		e1_pool = list(set([each[1] for each in ds_data]))
		e2_pool = list(set([each[2] for each in ds_data]))

		ds_dict = {}  # {e1_e2:[rel1,rel2],} 如果{'阿里巴巴_马云':[董事长，总裁]}
		for each in ds_data:
			relation, e1, e2 = each
			try:
				ds_dict[e1+'_'+e2].append(relation)
			except:
				ds_dict[e1+'_'+e2] = [relation]
		self.ds_data = ds_data
		self.all_relations = all_relations
		self.e1_pool = e1_pool
		self.e2_pool = e2_pool
		self.ds_dict = ds_dict

		logging.info('triple number: %d'%len(ds_data))
		logging.info('e1_pool number: %d'%len(e1_pool))
		logging.info('e2_pool number: %d'%len(e2_pool))
		logging.info('all relations number: %d' % len(all_relations))

		if os.path.exists(config.data_path + config.all_relations_file) is False:
			logging.info('将所有关系写入到\%s'%(config.data_path + config.all_relations_file))
			with open(config.extra_dict_path,'w') as f:
				for rel in all_relations:
					f.write(rel+'\n')

	def __find_entity(self,sentence,pool):
		'''
		如果ner的效果够好，可以换成ner的
		:param sentence:
		:param pool:
		:return: {position:entity}
		'''
		e_pool = {}
		for e in pool:
			for m in re.finditer(e, sentence):
				e_pos = m.start()
				if e_pos in e_pool:
					if len(e_pool[e_pos]) < len(e):
						e_pool[e_pos] = e
				else:
					e_pool[e_pos] = e
		true_pools = e_pool.copy()
		for pos1,e1 in e_pool.items():
			for pos2,e2 in e_pool.items():
				if e1 == e2:
					continue
				if abs(pos1-pos2) < 10:
					if e2 in e1:
						del true_pools[pos2]
		return true_pools

	def __label_re(self,sentences,ds_dict,e1_pool,e2_pool,i):
		# 考虑到后续处理一致性，因此不分正负样标注
		logging.info('process %d start'%i)
		f = open(config.sample_path + 'sample' + str(i) + '.txt', 'w')
		for sent_id,sent in enumerate(sentences):
			if len(sent) < config.min_character or len(sent) > config.max_character:
				continue
			# 如果对于同一种类型，且两个pool相同，只用find_entity一次
			if sent_id%1000 == 0:
				logging.info('process %d processed sentence %d'%(i,sent_id))
			# 适当调整找e1和e2的顺序进行加速
			e1_dict = self.__find_entity(sent, e1_pool)
			if len(e1_dict) == 0:
				logging.debug('e1:%s' % sent)
				continue
			e2_dict = self.__find_entity(sent, e2_pool)
			if len(e2_dict) == 0:
				logging.debug('e2:%s' % sent)
				continue

			# 如果句子中相同给的entity出现次数多，会使得效率低
			for pos1,e1 in e1_dict.items():
				for pos2,e2 in e2_dict.items():
					if pos1 == pos2: # 王志集团有限公司 王志
						continue
					logging.debug('%s\t%d:%s\t%d:%s' % (sent,pos1,e1,pos2,e2))
					try:
						relations = ds_dict[e1+'_'+e2]
						f.write(
							'|'.join(relations) + '\t' + sent + '\t' + e1 + '\t' + e2 + '\t' + str(pos1) + '\t' + str(
								pos2) + '\n')
					except:
						# kg中没有关系的标注为'未知关系'
						f.write('未知关系' + '\t' + sent + '\t' + e1 + '\t' + e2 + '\t' + str(pos1) + '\t' + str(pos2) + '\n')
		f.close()


	def __label_ner(self,sentences,ds_data,i):
		pass

	def __label_joint(self,sentences,ds_data,i):
		pass

	def label(self, sentences, task, process_num = 8):
		'''

		:param sentences:
		:param task:
		:param process_num: 进程数
		:return:
		'''
		epoch_size = int((len(sentences) + process_num - 1) / process_num)
		# epoches = int((len(sentences) + epoch_size - 1) / epoch_size)
		if task == 're':
			logging.info('开始为关系抽取进行数据标注')
			for i in range(process_num):
				sents = sentences[i*epoch_size:i*epoch_size+epoch_size]
				p = Process(target=self.__label_re, args=(sents,self.ds_dict,self.e1_pool,self.e2_pool,i))
				p.start()
		elif task == 'ner':
			logging.warning('label ner待完善')
		elif task == 'joint':
			logging.warning('label joint待完善')

	def __filter_re(self,data):
		'''
		过滤初始标记的数据，规则需要根据远程监督标记出的数据出现的问题具体编写
		目前提供的是「人与公司」两个规则，是否关键字过滤已标记的有关系的数据？规则过滤'未知关系'的数据？
		:param data：[relations,sent,e1,e2,pos1,pos2]，relations有可能是'未知关系'
		:param save_name default filter.txt
		'''
		import jieba
		jieba.load_userdict(config.extra_dict_path)
		# all_doing_relation = self.all_relations
		save_name = 'filter.txt'
		with open(config.sample_path + save_name,w) as f:
			for each in data:
				relations,sent,e1,e2,pos1,pos2 = each
				true_relations = []
				if relations == '未知关系':
					if abs(pos1 - pos2) > 80:
						relations = '0'
					elif sent.count('，') > 4 and '，' in sent[pos1:pos2]:
						relations = '0'
					elif sent.count('、') > 4 and '、' in sent[pos1:pos2]:
						relations = '0'
					elif '；' in sent[pos1:pos2]:
						relations = '0'
					f.write(relations + '\t' + sent + '\t' + e1 + '\t' + e2 + '\t' + str(pos1) + '\t' + str(
						pos2) + '\n')
				else:
					seg_sent = ' '.join(jieba.cut(sent))
					relations = relations.split('|')
					for rel in relations:
						if rel in seg_sent:
							# 可以在加些距离限制包括两实体距离，两实体跟距离关键词的位置
							true_relations.append(rel)
					f.write(
						'|'.join(true_relations) + '\t' + sent + '\t' + e1 + '\t' + e2 + '\t' + str(pos1) + '\t' + str(
							pos2) + '\n')

	def __filter_ner(self,):
		'''
		'''
		pass

	def filter_labeled_data(self,data,task):
		if task == 're':
			self.__fiter_re(data)
		'''
		规则过滤样本
		'''
		pass

	def __replace_entity(self,sent,e1,e2,pos1,pos2):
		# 用'E1'和'E2'替换「指定位置」的原实体，sent是没有分词的句子
		# yzq,写好了之后更新下__gen_train_re里面的sent
		ret = []
		for w in sent:
			ret.append(w)
		ret[pos1] = 'E1'
		for i in range(len(e1)-1):
			ret[pos1+i+1] = ''
		ret[pos2] = 'E2'
		for i in range(len(e2)-1):
			ret[pos2+i+1] = ''
		ret = ' '.join(ret)
		return  ret

	def __gen_train_re(self,data,binary,train_ratio,need_ner = False):
		'''
		为模型的训练生成数据
		:param data: [relations,sent,e1,e2,pos1,pos2]
		:param binary: 是否是多个二分类模型
		:return:
		'''
		train_data = pd.DataFrame()
		test_data = pd.DataFrame()
		if binary:
			rels, sents, e1s, e2s, pos1s, pos2s = []*6
			for each in data:
				relations, sent, e1, e2, pos1, pos2 = each
				sent = self.__replace_entity(sent,e1,e2,pos1,pos2)
				if relations == '未知关系':
					continue
				relations = relations.split('|')
				for rel in relations:
					rels.append(rel)
					sents.append(sent)
					e1s.append(e1)
					e2s.append(e2)
					pos1s.append(pos1)
					pos2s.append(pos2)
			data = pd.DataFrame(
				{'relations': rels, 'sentences': sents, 'entity1': e1s, 'entity2': e2s, 'pos1': pos1s,
				 'pos2': pos2s})

			no_rel_data = data[data.relations=='0']
			rel_data = data[data.relations != '0']
			neg_indices = np.random.permutation(np.arange(no_rel_data.shape[0]))
			pos_indices = np.random.permutation(np.arange(rel_data.shape[0]))

			train_pos_num = int(rel_data.shape[0] * train_ratio)
			train_neg_num = int(no_rel_data.shape[0] * train_ratio)

			train_data = train_data.append(rel_data.iloc[pos_indices[0:train_pos_num]], ignore_index=True)
			train_data = train_data.append(no_rel_data.iloc[neg_indices[0:train_neg_num]], ignore_index=True)

			test_data = test_data.append(rel_data.iloc[pos_indices[train_pos_num:]], ignore_index=True)
			test_data = test_data.append(no_rel_data.iloc[neg_indices[train_neg_num:]], ignore_index=True)
		else:
			rels, sents, e1s, e2s, pos1s, pos2s = [] * 6
			for each in data:
				relations, sent, e1, e2, pos1, pos2 = each
				if relations == '未知关系':
					continue
				rels.append(relations)
				sents.append(sent)
				e1s.append(e1)
				e2s.append(e2)
				pos1s.append(pos1)
				pos2s.append(pos2)
			data = pd.DataFrame(
				{'relations': rels, 'sentences': sents, 'entity1': e1s, 'entity2': e2s, 'pos1': pos1s,
				 'pos2': pos2s})
			indices = np.random.permutation(np.arange(data.shape[0]))
			train_num = int(data.shape[0] * train_ratio)

			train_data.append(data.iloc[indices[0:train_num]], ignore_index=True)
			test_data.append(data.iloc[indices[train_num:]], ignore_index=True)
		train_data.to_csv(config.data_path + config.train_file, sep='\t', index=False, encoding='utf-8')
		test_data.to_csv(config.data_path + config.test_file, sep='\t', index=False, encoding='utf-8')

	# 应该放在 data.py中的，如果是在这里可能需要进行ner
	def gen_train_sample(self,data,task,binary,train_ratio,need_ner=False):
		if task == 're':
			self.__gen_train_re(data,binary,train_ratio,need_ner)


if __name__ == '__main__':
	task = config.task
	original_file_name = 'original_text.txt'
	with open(config.data_path + original_file_name) as f:
		sentences = [line.strip() for line in f]

	sample = Sample(src_path=config.data_path + config.ds_file)
	sample.label(sentences[::-1],task=task)

	# 执行完之后命令行中运行
	with open(config.sample_path + 'sample.txt',w) as f:
		data = [line.strip().split('\t') for line in f]
	sample.filter_labeled_data(data,task=task)

	with open(config.sample_path + 'filter.txt') as f:
		data = [line.strip().split('\t') for line in f]

	need_ner = False
	if 'ner' in config.model:
		need_ner = True
	sample.gen_train_sample(data,binary=config.binary,task=task,train_ratio=config.train_ratio,need_ner=need_ner)
