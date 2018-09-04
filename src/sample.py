# coding=UTF-8

def _label_re(sentences,kg_data):
	'''
	采样 == 为relation extract task 进行 DS标注
	多线程，会放在不同文件中，(计时)

	'''
	pass

def _label_ner(sentences,kg_data):
	'''
	通过DS 为ner task进行标注
	'''
	pass

def _label_joint(sentences,kg_data):
	'''
	'''
	pass

def label(task = 're', sentences, kg_data):
	if task == 're':
		_label_re(sentences,kg_data)
	# ...
	pass

def _filter_re():
	'''
	过滤relation extract的数据
	'''
	pass

def _filter_ner():
	'''
	'''
	pass

def filter_labeled_data(task = 're'):
	'''
	规则过滤样本
	'''
	pass


def gen_train_sample(task = 're', train_ratio = 0.75):
	'''
	正负样本生成训练集
	'''
	pass

if __name__ == '__main__':
	# 上述三个函数
	# label()
	# filter_labeled_data()
	# gen_train_sample()
	pass