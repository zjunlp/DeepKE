# coding=utf-8
import config
import pandas as pd
from collections import Counter
import pdb
import numpy as np
import tensorflow as tf
import time
import evaluation
import os
import codecs
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
	exit()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID  # "0, 1" for multiple
configDevice = tf.ConfigProto()
configDevice.gpu_options.allow_growth = True

if config.which_relation == 'per':
	top_relation = config.per_com_relation
else:
	print('relation输入错误，在config.py中更改')
	exit()


def evaluate(sess, model, data):
	# sents, masks, relations, e1, e2, dist1, dist2, ner = data
	# in_bz, in_masks, in_x, in_e1, in_e2, in_dist1, in_dist2, in_ner, in_y = model.inputs
	sents, masks, relations, e1, e2, rel, dist1, dist2, ner = data
	in_bz, in_masks, in_x, in_e1, in_e2, in_rel, in_dist1, in_dist2, in_ner, in_y = model.inputs
	# sents, masks, relations, e1, e2, dist1, dist2, pos, ner = data
	# in_bz, in_masks, in_x, in_e1, in_e2, in_dist1, in_dist2, in_pos, in_ner, in_y = model.inputs
	batch_size = 500
	res = []
	acc_sum = 0
	for i in range(len(sents) // batch_size + 1):
		feed_dict = {in_bz: min(batch_size, len(sents) - batch_size * i),
					 in_masks: masks[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_x: sents[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_e1: e1[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_e2: e2[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_dist1: dist1[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_dist2: dist2[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 # in_pos: pos[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_rel: rel[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_ner: ner[batch_size * i:min(batch_size * (i + 1), len(sents))],
					 in_y: relations[batch_size * i:min(batch_size * (i + 1), len(sents))]}
		acc, preds = sess.run([model.accuracy, model.preds], feed_dict=feed_dict)
		acc_sum += acc * min(batch_size, len(sents) - batch_size * i)
		res.append(preds)
	preds = np.concatenate(res)
	acc = acc_sum / float(len(sents))

	micro_f1, macro_f1 = evaluation.evaluate(preds, relations)
	p, r, f1 = evaluation.evaluate_micro_p_r_f1(preds, relations)

	# 这里实现macro-average
	return acc, micro_f1, macro_f1, p, r, preds, relations


def run_epoch(session, model, batch_iter, is_training=True, verbose=True):
	start_time = time.time()
	acc_count = 0
	step = 0  # len(all_data)

	total = 0
	for batch in batch_iter:
		total += len(batch)
		step += 1
		batch = (x for x in zip(*batch))

		# sents,masks,relations, e1, e2, dist1, dist2, pos, ner = batch
		# sents,masks,relations, e1, e2, dist1, dist2, ner = batch
		sents, masks, relations, e1, e2, rel, dist1, dist2, ner = batch

		# sents is a list of np.ndarray, convert it to a single np.ndarray
		sents = np.vstack(sents)

		in_bz, in_masks, in_x, in_e1, in_e2, in_rel, in_dist1, in_dist2, in_ner, in_y = model.inputs
		feed_dict = {in_bz: config.BATCH_SIZE, in_masks: masks, in_x: sents, in_e1: e1, in_e2: e2, in_rel: rel,
					 in_dist1: dist1,
					 in_dist2: dist2, in_ner: ner, in_y: relations}
		# in_bz,in_masks,in_x, in_e1, in_e2, in_dist1, in_dist2, in_ner, in_y = model.inputs
		# feed_dict = {in_bz: config.BATCH_SIZE,in_masks:masks, in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1,
		# 			 in_dist2: dist2, in_ner: ner, in_y: relations}
		# in_bz,in_masks,in_x, in_e1, in_e2, in_dist1, in_dist2, in_pos, in_ner, in_y = model.inputs
		# feed_dict = {in_bz: config.BATCH_SIZE,in_masks:masks, in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1,
		# 			 in_dist2: dist2, in_pos: pos, in_ner: ner, in_y: relations}
		if is_training:
			_, _, acc, loss = session.run([model.train_op, model.reg_op, model.acc, model.loss], feed_dict=feed_dict)
			acc_count += acc
			if verbose and step % 10 == 0:
				tf.logging.info("  step: %d acc: %.2f%% loss: %.2f time: %.2f" % (
					step,
					acc_count / (step * config.BATCH_SIZE) * 100,
					loss,
					time.time() - start_time
				))
		else:
			acc, loss = session.run([model.acc, model.loss], feed_dict=feed_dict)
			acc_count += acc
	# 感觉这儿有点问题
	return acc_count / float(total), loss


def build_dict(sentences):
	# 统计每一个单词出现的次数
	word_count = Counter()
	for s in sentences:
		for w in s.split():
			word_count[w] += 1
	vocabs = word_count.most_common()
	word_dict = {w[0]: i + 1 for (i, w) in enumerate(vocabs)}

	with open(config.DATA_DIR + 'word_dict.txt', 'w') as ofile:
		for w in word_dict:
			ofile.write(str(word_dict[w]) + '\t' + w + '\t' + str(word_count[w]) + '\n')
	return word_dict


def build_ner_dict(sentences):
	word_count = Counter()
	for s in sentences:
		for w in s.split():
			word_count[w] += 1
	vocabs = word_count.most_common()
	word_dict = {w[0]: i + 1 for (i, w) in enumerate(vocabs)}

	with open(config.DATA_DIR + 'ner_dict.txt', 'w') as ofile:
		for w in word_dict:
			ofile.write(str(word_dict[w]) + '\t' + w + '\t' + str(word_count[w]) + '\n')
	return word_dict


def build_embedding(word_dict):
	embedding_dict = {}
	with codecs.open(config.EMBEDDING_DIR + 'words.lst', 'r', encoding='utf-8') as f:
		for i, l in enumerate(f.readlines()):
			embedding_dict[l.strip().lower()] = i

	with codecs.open(config.EMBEDDING_DIR + 'embeddings.txt', 'r', encoding='utf-8') as f:
		embeddings = f.readlines()

	num_words = len(word_dict) + 1
	dim = config.EMBEDDING_DIM
	# 保存embedding向量
	embedding_matrix = np.random.uniform(-0.01, 0.01, size=(num_words, dim))
	common_words = 0
	for w in word_dict:
		if w in embedding_dict:
			embedding_matrix[word_dict[w]] = [float(x) for x in embeddings[embedding_dict[w]].strip().split()]
			common_words += 1
	embedding_matrix[0] = np.zeros((dim))
	print('embeddings: %d, valid: %d' % (num_words, common_words))
	return embedding_matrix.astype(np.float32)


def load_dict(f):
	word_dict = {}
	with open(config.DATA_DIR + f, 'r', ) as f:
		for l in f.readlines():
			tmp = l.strip().split('\t')
			word_dict[tmp[1]] = tmp[0]
	return word_dict


def convert_to_sequence(data, word_dict, classid):
	word_dict = load_dict('word_dict.txt')
	# pos_dict = load_dict('pos_dict.txt')
	ner_dict = load_dict('ner_dict.txt')

	num_samples = data.shape[0]
	sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	pos1_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	pos2_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	# pos_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)
	ner_sequences = np.zeros((num_samples, config.MAX_LEN), dtype=int)

	entity1 = []
	entity2 = []

	context = []
	masks = []
	rel = [word_dict[top_relation[classid - 1]]] * num_samples
	# for i, (sent, e11, e21, pos, ner, ctx, similarity) in enumerate(zip(data.sentences, data.entity1_b, data.entity2_b, data.pos, data.ner, data.ctx, data.sim)):
	# for i, (sent, e11, e21, sos, ner) in enumerate(
	# 		zip(data.sentences, data.entity1_b, data.entity2_b, data.pos, data.ner)):
	for i, (sent, e11, e21, ner) in enumerate(
			zip(data.sentences, data.entity1_b, data.entity2_b, data.ner)):
		# sent_splitted = pos.split()
		e11, e21 = int(e11), int(e21)
		head_index = e11
		tail_index = e21

		# tmp = [pos_dict[w] if w in pos_dict else 0 for w in sent_splitted]
		# tmp = tmp[:config.MAX_LEN]
		# pos_sequences[i, :len(tmp)] = tmp

		# convert ner to id, and padding
		sent_splitted = ner.split()
		tmp = [ner_dict[w] if w in ner_dict else 0 for w in sent_splitted]
		tmp = tmp[:config.MAX_LEN]
		ner_sequences[i, :len(tmp)] = tmp

		# convert words to id, and padding
		sent_splitted = sent.split()
		tmp = [word_dict[w] if w in word_dict else 0 for w in sent_splitted]
		tmp = tmp[:config.MAX_LEN]
		sequences[i, :len(tmp)] = tmp

		pos1_sequences[i, :] = [func(idx - e11) + config.MAX_LEN for idx in range(config.MAX_LEN)]
		pos2_sequences[i, :] = [func(idx - e21) + config.MAX_LEN for idx in range(config.MAX_LEN)]

		e1 = []
		e2 = []
		if e11 == 0:
			e1.append(word_dict[sent_splitted[e11]])
		else:
			e1.append(word_dict[sent_splitted[e11 - 1]])

		if e21 == 0:
			e2.append(word_dict[sent_splitted[e21]])
		else:
			e2.append(word_dict[sent_splitted[e21 - 1]])

		# e2.append(word_dict[sent_splitted[e21 - 1]])
		e1.append(word_dict[sent_splitted[e11]])
		e2.append(word_dict[sent_splitted[e21]])

		if e11 == (len(sent_splitted) - 1):
			e1.append(word_dict[sent_splitted[e11]])
		else:
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

	return sequences, masks, data.relations, entity1, entity2, rel, pos1_sequences, pos2_sequences, ner_sequences,


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
	if n < -config.MAX_LEN + 1:
		return -config.MAX_LEN + 1
	elif n > config.MAX_LEN - 1:
		return config.MAX_LEN - 1
	else:
		return n


def batch_iter(data, batch_size, shuffle=True):
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


def get_DS_neg(train_neg, pos_num):
	neg_num = train_neg.shape[0]
	half_pos_num = int(pos_num / 2)
	cur1_all_num = int(pos_num * 0.5)
	# cur2_all_num = int(pos_num*0.5)
	cur2_all_num = pos_num * 2
	cur1_num = 0
	cur2_num = 0
	candidate_list = np.random.permutation(np.arange(neg_num))
	sss = []
	choose_list = []
	for i in candidate_list:
		cur_data = train_neg.iloc[i]
		s = cur_data.sentences
		if s in sss or ' 董事 E2' in s or 'E2 董事 ' in s:
			continue
		sss.append(s)
		e2 = cur_data.entity2
		if ' 董事 ' in s:
			continue
		elif cur2_all_num > cur2_num:
			choose_list.append(i)
			cur2_num += 1
		if cur1_num + cur2_num > 2 * pos_num:
			break
	print('获取负样本%d' % (cur1_num + cur2_num))
	ret = train_neg.iloc[choose_list]
	ret.to_csv('../data/dongshi_train.csv', sep='\t', index='ignore')
	return train_neg.iloc[choose_list[0:pos_num]]


def main(_):
	train_data = pd.read_csv(config.TRAIN_FILE, sep='\t')
	test_data = pd.read_csv(config.TEST_FILE, sep='\t')

	print('train class number pos : ' + ",".join(
		str(train_data[train_data.relations == i].shape[0]) for i in range(config.CLASS_NUM)))
	print('test class number pos: ' + ','.join(
		str(test_data[test_data.relations == i].shape[0]) for i in range(config.CLASS_NUM)))

	# build vocab
	word_dict = build_dict(train_data['sentences'].tolist() + test_data['sentences'].tolist())
	print('total words: %d' % len(word_dict))
	ner_dict = build_ner_dict(train_data['ner'].tolist() + test_data['ner'].tolist())
	print('total ner: %d' % len(ner_dict))

	# build embedding
	embeddings = build_embedding(word_dict)
	specific = config.specific
	if specific is None:
		specific = range(1, config.CLASS_NUM)
	record_class_f1 = {}

	train_neg = train_data[train_data.relations == 0]  # 因为各个职位之间可能有关系
	columns1 = ['relations', 'sentences', 'entity1', 'entity2', 'entity1_b', 'entity1_e', 'entity2_b', 'entity2_e','ner']
	columns2 = ['preds', 'relations', 'sentences', 'entity1', 'entity2', 'entity1_b', 'entity1_e', 'entity2_b','entity2_e']

	for classid in specific:
		relation = top_relation[classid - 1]
		print('\nclass %d: %s' % (classid, relation))
		tmp_train_pos = train_data[train_data.relations == classid]
		if tmp_train_pos.shape[0] > 5000: # 样本过多，随机选择5000句训练
			tmp_train_pos = tmp_train_pos.iloc[np.random.permutation(np.arange(5000))]
		tmp_train_neg = pd.read_csv(config.RESULT_DIR + relation + '.csv', sep='\t')

		'''
		# 负样本1：随机从所有负样本中选择与正样本等量的负样本进行训练，需要进行多伦的随机选择
		if relation == '董事':  # 确保董事能有一些高质量的负样本
			tmp_train_neg = get_DS_neg(train_neg, tmp_train_pos.shape[0])
		else:
			tmp_train_neg = train_neg.iloc[np.random.permutation(np.arange(tmp_train_pos.shape[0]))]
		tmp_train_neg.to_csv(config.DATA_DIR + relation + '.csv', sep='\t', columns=columns1, index='ignore')
		'''
		# 负样本2：在data下已经保存了一批比较高质量的负样本，命名格式如 董事.csv，直接读取
		tmp_train_neg = pd.read_csv(config.DATA_DIR + relation + '.csv', sep='\t')
		tmp_train_data = tmp_train_pos.append(tmp_train_neg)
		tmp_train_data.loc[:, 'relations'] = tmp_train_data['relations'].apply(lambda x: 1 if x == classid else 0)
		tmp = tmp_train_data['relations'].sum()
		print("---------Train Postive: %d, Negative: %d" % (tmp, tmp_train_data.shape[0] - tmp))
		record_class_f1[classid] = [str(tmp), str(tmp_train_data.shape[0] - tmp)]

		tmp_test_pos = test_data[test_data.relations == classid]
		tmp_test_neg = test_data[test_data.relations == 0]
		tmp_test_neg = tmp_test_neg[0:tmp_test_pos.shape[0]]
		tmp_test_data = tmp_test_pos.append(tmp_test_neg)
		tmp_test_data.loc[:, 'relations'] = test_data['relations'].apply(lambda x: 1 if x == classid else 0)
		tmp = tmp_test_data['relations'].sum()
		print("---------Test Postive: %d, Negative: %d" % (tmp, tmp_test_data.shape[0] - tmp))
		record_class_f1[classid].extend([str(tmp), str(tmp_test_data.shape[0] - tmp)])

		train_seq = convert_to_sequence(tmp_train_data, word_dict, classid)
		test_seq = convert_to_sequence(tmp_test_data, word_dict, classid)

		# Train
		with tf.Graph().as_default():
			with tf.name_scope("Train"):
				with tf.variable_scope("Model", reuse=None):
					model_train = Model(embeddings, config.BATCH_SIZE, is_training=True)

			with tf.name_scope("Valid"):
				with tf.variable_scope("Model", reuse=True):
					model_test = Model(embeddings, tmp_test_data.shape[0], is_training=False)

			saver = tf.train.Saver()
			sv = tf.train.Supervisor(global_step=model_train.global_step)
			best_macro_f1 = 0
			best_p = 0
			best_r = 0
			best_preds = None
			best_epoch = 0
			with sv.managed_session(config=configDevice) as session:
				for epoch in range(config.EPOCHES):
					train_iter = batch_iter(list(zip(*train_seq)), config.BATCH_SIZE, shuffle=True)
					# test_iter = batch_iter(list(zip(*test_seq)), config.BATCH_SIZE, shuffle=False)
					train_acc, loss = run_epoch(session, model_train, train_iter, verbose=False)
					test_acc, micro_f1, macro_f1, p, r, preds, labels = evaluate(session, model_test, test_seq)
					if micro_f1 > best_macro_f1:
					# if p > best_p:
						best_macro_f1 = micro_f1
						best_p = p
						best_r = r
						saver.save(session, config.MODEL_PATH + '_' + str(classid) + '_model')
						best_preds = preds
						best_epoch = epoch

					print(
						"Epoch: %d Train: %.2f%%  loss: %.2f  Test: %.2f%% Micro-F1: %.2f%% Precision: %.2f%% Recall: %.2f%%" % (
							epoch + 1, train_acc * 100, loss, test_acc * 100, micro_f1 * 100, p * 100, r * 100))
					if loss < 0.0001 or test_acc > 0.965:
						break

			print('best_macro_f1 %d : %0.02f%%\n' % (best_epoch, best_macro_f1 * 100))
			record_class_f1[classid].extend(
				[str(best_epoch), str(best_macro_f1 * 100), str(best_p * 100), str(best_r * 100)])

		# 保存对test的预测结果
		data = tmp_test_data
		data['preds'] = best_preds

		tmp_save_path = config.TRAIN_SAVE_PATH + '_' + str(classid) + '.csv'
		data.to_csv(tmp_save_path, index=False, sep='\t',columns=columns2)
		print('save to : ' + tmp_save_path)

	print('classid: \t train_pos\ttrain_neg\ttest_pos\ttest_neg\tbest_f1\tbest_p\tbest_r')
	for classid, record in record_class_f1.items():
		print(top_relation[classid - 1] + ' : ' + '\t'.join(record))


if __name__ == '__main__':
	tf.app.run()
