# coding=utf-8 
import config
import pandas as pd
import numpy as np
import tensorflow as tf
import os

if config.which_model == 'cnn':
	from cnn_model import Model

	print('using model cnn')
elif config.which_model == 'bilstm':
	from bilstm_model import Model

	print('using model bilstm')
elif config.which_model == 'pcnn':
	from pcnn_model import Model

	print('using model pcnn')
else:
	print('config.model 未定义')
	exit(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID  # "0, 1" for multiple
configDevice = tf.ConfigProto()
configDevice.gpu_options.allow_growth = True
top_relation = config.per_com_relation


def convert_to_sequence(data, classid):
	word_dict = load_dict(config.DATA_DIR + 'word_dict.txt')
	# pos_dict = load_dict('pos_dict.txt')
	ner_dict = load_dict(config.DATA_DIR + 'ner_dict.txt')

	num_samples = data.shape[0]
	sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	pos1_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	pos2_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	# pos_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	ner_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)

	entity1 = []
	entity2 = []
	masks = []
	rel = [word_dict[top_relation[classid - 1]]] * num_samples
	# for i, (sent, e11, e21, pos, ner) in enumerate(
	# 		zip(data.sentences, data.entity1_b, data.entity2_b, data.pos, data.ner)):
	for i, (sent, e11, e21, ner) in enumerate(
			zip(data.sentences, data.entity1_b, data.entity2_b, data.ner)):
		# sent_splitted = pos.split()
		# tmp = [pos_dict[w] if w in pos_dict else 0 for w in sent_splitted]
		# tmp = tmp[:config.MAX_LEN]
		# pos_sequences[i, :len(tmp)] = tmp

		head_index = int(e11)
		tail_index = int(e21)

		# convert ner to id, and padding
		ner_splitted = ner.split()
		for word in ner_splitted:
			if word not in ner_dict:
				ner_dict[word] = len(ner_dict) + 1
		tmp = [ner_dict[w] if w in ner_dict else 0 for w in ner_splitted]
		tmp = tmp[:config.MAX_LEN]
		ner_sequences[i, :len(tmp)] = tmp

		sent_splitted = sent.split()
		# 对于新的单词的处理
		for word in sent_splitted:
			if word not in word_dict:
				word_dict[word] = len(word_dict) + 1
		tmp = [word_dict[w] if w in word_dict else 0 for w in sent_splitted]
		tmp = tmp[:config.MAX_LEN]
		sequences[i, :len(tmp)] = tmp

		pos1_sequences[i, :] = [func(idx - e11) + config.MAX_LEN for idx in range(config.MAX_LEN)]
		pos2_sequences[i, :] = [func(idx - e21) + config.MAX_LEN for idx in range(config.MAX_LEN)]

		'''
		tmp = tmp[-config.MAX_LEN:] #限制句子最大长度
		sequences[i,-len(tmp):] = tmp  #padding,注意这里是在句
		'''

		'''
		entity1.append(word_dict[sent_splitted[e11]])
		entity2.append(word_dict[sent_splitted[e21]])
		'''
		e1 = []
		e2 = []
		if e11 == 0:
			e1.append(word_dict[sent_splitted[e11]])
		else:
			e1.append(word_dict[sent_splitted[e11 - 1]])
		e2.append(word_dict[sent_splitted[e21 - 1]])
		e1.append(word_dict[sent_splitted[e11]])
		e2.append(word_dict[sent_splitted[e21]])
		e1.append(word_dict[sent_splitted[e11 + 1]])
		if e21 == (len(sent_splitted) - 1):
			e2.append(word_dict[sent_splitted[e21]])
		else:
			e2.append(word_dict[sent_splitted[e21 + 1]])
		entity1.append(e1)
		entity2.append(e2)

		if head_index > tail_index:
			head_index, tail_index = tail_index, head_index
		if tail_index >= config.MAX_LEN:
			tail_index = config.MAX_LEN - 1
		if head_index >= config.MAX_LEN:
			head_index = config.MAX_LEN - 1

		if head_index == tail_index:  # 都是89
			head_index = tail_index - 1

		mask = pcnn_mask(len(sent_splitted), head_index, tail_index)
		masks.append(mask)

	# return sequences,masks, data.relations, entity1, entity2, pos1_sequences, pos2_sequences, pos_sequences, ner_sequences,
	return sequences, masks, data.relations, entity1, entity2, rel, pos1_sequences, pos2_sequences, ner_sequences,


def load_dict(f):
	word_dict = {}
	with open(config.DATA_DIR + f, 'r') as f:
		for l in f.readlines():
			tmp = l.strip().split('\t')
			word_dict[tmp[1]] = tmp[0]
	return word_dict


def pcnn_mask(length, b, d):
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


def func(n):
	if (n < -config.MAX_LEN + 1):
		return -config.MAX_LEN + 1
	elif (n > config.MAX_LEN - 1):
		return config.MAX_LEN - 1
	else:
		return n


class Model(object):
	def __init__(self, model):
		self.inputs = model['inputs']
		self.accuracy = model['accuracy']
		self.preds = model['preds']


def get_variable_from_graph(graph):
	model = {}
	model['inputs'] = []
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_bz:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_masks:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_x:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_e1:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_e2:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_rel:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_dist1:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_dist2:0'))
	# model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_pos:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_ner:0'))
	model['inputs'].append(graph.get_tensor_by_name('Valid/Model/in_y:0'))
	model['accuracy'] = graph.get_tensor_by_name('Valid/Model/Mean:0')
	model['preds'] = graph.get_tensor_by_name('Valid/Model/Cast:0')
	return Model(model)


def evaluate():
	all_predict = pd.DataFrame()
	specific = config.specific
	if specific == None:
		specific = range(1, config.CLASS_NUM)
	for classid in specific:
		rel = top_relation[classid - 1]
		data = pd.read_csv(config.PREDICT_SAVE_PATH + '_' + str(classid) + '.csv', sep='\t')
		# data.loc[data.preds==rel]=1 # preds == rel replace with preds = 1
		sentences = data.sentences
		sentences = set(sentences)
		new_data = pd.DataFrame()
		for sent in sentences:
			flag = True
			tmp_data = data[data.sentences == sent]
			if tmp_data.shape[0] == 1:
				new_data = new_data.append(tmp_data, ignore_index=True)
				continue
			pred1 = tmp_data[tmp_data.preds == 1]
			if pred1.shape[0] > 0:
				new_data = new_data.append(pred1.iloc[0], ignore_index=True)
			else:
				new_data = new_data.append(tmp_data.iloc[0], ignore_index=True)

		preds1 = new_data[new_data.preds == 1]
		relation1 = new_data[new_data.relations == 1]
		a = preds1[preds1.relations == 1]
		b = relation1[relation1.preds == 1]
		p = a.shape[0] / float(preds1.shape[0])
		r = b.shape[0] / float(relation1.shape[0])
		if p + r == 0:
			f = 0
		else:
			f = 2 * p * r / (p + r)
		print('class%d:\t%s:\t%d\tPrecision: %.2f%% Recall: %.2f%% Micro-F1: %.2f%% ' % (
			classid, top_relation[classid - 1], relation1.shape[0], p * 100, r * 100, f * 100))
		new_data = new_data.replace({'relations': 1, 'preds': 1}, rel)
		new_data = new_data
		all_predict = all_predict.append(new_data)

	all_predict[['entity1_b', 'entity1_e', 'entity2_b', 'entity2_e']].apply(lambda x: map(int, x))
	all_predict = all_predict.replace({'relations': '0.0', 'preds': '0.0'}, 0)
	# label0 = all_predict[all_predict.relations == 0]
	# 每个相同为0的句子，被判断为0 > 5 次，则为
	# def f(x):
	# 	x.entity1_b = int(x.entity1_b)
	# 	x.entity1_e = int(x.entity1_e)
	# 	x.entity2_b = int(x.entity2_b)
	# 	x.entity2_e = int(x.entity2_e)

	diff = all_predict[all_predict.relations != all_predict.preds]
	same = all_predict[all_predict.relations == all_predict.preds]
	all_predict = same.append(diff)
	all_predict[['entity1_b', 'entity1_e', 'entity2_b', 'entity2_e']].applymap(lambda x: int(x))
	all_predict.replace({'relations': 1, 'preds': 1}, rel)
	columns = ['preds', 'relations', 'sentences', 'entity1', 'entity2', 'entity1_b', 'entity1_e', 'entity2_b',
			   'entity2_e']
	all_predict.to_csv(config.PREDICT_SAVE_PATH + '.csv', sep='\t', index=False, columns=columns)


def test():
	test_data = pd.read_csv(config.DATA_DIR + config.PREDICT_FILE_NAME.split('.')[0] + '.csv', sep='\t')

	specific = config.specific
	if specific is None:
		specific = range(1, config.CLASS_NUM)

	for classid in specific:
		relation = top_relation[classid - 1]
		tmp_test_pos = test_data[test_data.relations == relation]
		tmp_test_neg = test_data[test_data.relations == '0']
		pos_num = tmp_test_pos.shape[0]
		print('%s:%d' % (relation, pos_num))
		tmp_test_pos.relations = [1] * pos_num
		tmp_test_neg.relations = [0] * tmp_test_neg.shape[0]
		tmp_test_data = tmp_test_pos.append(tmp_test_neg)
		total_num = tmp_test_data.shape[0]

		test_seq = convert_to_sequence(tmp_test_data, classid)
		sess = tf.Session()
		preds_all = []
		# saver = tf.train.import_meta_graph(config.MODEL_PATH + '.meta')
		saver = tf.train.import_meta_graph(config.MODEL_PATH + '_' + str(classid) + '_model.meta')
		saver.restore(sess, config.MODEL_PATH + '_' + str(classid) + '_model')
		# saver.restore(sess, tf.train.latest_checkpoint('./'))
		graph = tf.get_default_graph()
		model = get_variable_from_graph(graph)

		# sents,masks, relations, e1, e2, dist1, dist2,pos,ner = test_seq
		# in_bz,in_masks,in_x, in_e1, in_e2, in_dist1, in_dist2, in_pos, in_ner, in_y = model_test.inputs
		# feed_dict = {in_bz: len(sents), in_masks:masks ,in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1,
		# 						 in_dist2: dist2, in_pos:pos, in_ner:ner, in_y: relations}
		sents, masks, relations, e1, e2, rel, dist1, dist2, ner = test_seq
		in_bz, in_masks, in_x, in_e1, in_e2, in_rel, in_dist1, in_dist2, in_ner, in_y = model.inputs
		feed_dict = {in_bz: len(sents), in_masks: masks, in_x: sents, in_e1: e1, in_e2: e2, in_rel: rel,
					 in_dist1: dist1,in_dist2: dist2, in_ner: ner, in_y: relations}
		acc, preds = sess.run([model.accuracy, model.preds], feed_dict=feed_dict)
		new_preds = preds
		new_data = pd.DataFrame({'preds': new_preds,
			 'relations': tmp_test_data.relations,
			 'sentences': tmp_test_data.sentences,
			 'entity1': tmp_test_data.entity1,
			 'entity2': tmp_test_data.entity2,
			 'entity1_b': tmp_test_data.entity1_b,
			 'entity1_e': tmp_test_data.entity1_e,
			 'entity2_b': tmp_test_data.entity2_b,
			 'entity2_e': tmp_test_data.entity2_e})
		columns = ['preds', 'relations', 'sentences', 'entity1', 'entity2', 'entity1_b', 'entity1_e', 'entity2_b',
				   'entity2_e']
		tmp_save_path = config.PREDICT_SAVE_PATH + '_' + str(classid) + '.csv'
		new_data.to_csv(tmp_save_path, sep='\t', index=False, columns=columns)
		print('preds result save to :' + tmp_save_path + '\n\n')
		sess.close()
	evaluate()


def main(_):
	test()


if __name__ == '__main__':
	tf.app.run()