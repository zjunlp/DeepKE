# coding=utf-8
from utility import DottableDict


class Config(object):
	'''
	初始的参数，可以写入这里，
	'''
	config = DottableDict()
	config.allowDotting()
	config.relations = ['法定代表人','董事长','董事','总经理','总裁','副总经理','副总裁','财务总监','监事','独立董事',]
	config.relations_attribute = {
		'法定代表人':
				{'synonym':['法人'],
				 'antonym':['独立董事']},
		'董事长':
			{'synonym': [],
			 'antonym': ['董事长候选人','前董事长',]},
		'董事':
			{'synonym': ['非独立董事','独立董事','董事长','副董事长','外部董事',],
			 'antonym': ['董事会秘书']},
		'总经理':
			{'synonym': ['ceo','执行总经理','首席执行官'],
			 'antonym': ['副总经理']},
		'总裁':
			{'synonym': [],
			 'antonym': ['独立董事']},
		'副总经理':
			{'synonym': [],
			 'antonym': ['总经理']},
		'副总裁':
			{'synonym': [],
			 'antonym': ['总裁']},
		'财务总监':
			{'synonym': [],
			 'antonym': ['生产总监','财务副总监']},
		'监事':
			{'synonym': ['监事长'],
			 'antonym': []},
		'独立董事':
			{'synonym': ['独董'],
			 'antonym': ['非独立董事']},

	}
	config.max_len = 180
	config.embedding_dim = 200
	config.dist_embedding_dim = 5
	config.ner_embedding_dim = 10
	config.pos_embedding_num = 2 * config.max_len
	config.learning_rate = 0.004
	config.keep_out = 1

	config.num_filters = 200
	config.padding = 'SAME'
	config.filter_sizes = [3,4,5]
	config.pooling_way = 'max' # ['max','avg']

	config.data_path = '../data/'
	config.sample_path = config.data_path + 'sample/'
	config.ds_file = 'kg_triple.txt'
	config.train_file = 'train.csv'
	config.test_file = 'test.csv'
	config.predict_file = 'predict.csv'

	config.min_character = 12
	config.max_character = 400

	config.all_relations_file = 'all_relations.txt'
	config.binary = True  # 是用多个二分类还是多分类
	config.model = 'pcnn' # ['cnn','pcnn','cnn_ner'..] model下的
	config.train_ratio = 0.75
	config.task = 're'  # ['re','ner','joint'] 暂时定义，可能更改
	config.segment_way = 'jieba'  # ['jieba','character']
	config.ner_way = 'zju' # ['zju','fudan','self']

