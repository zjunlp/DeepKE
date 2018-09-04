# coding=utf-8
import tensorflow as tf
from config import Config
config = Config.config

class Encoder(object):
	def __init__(self,):
		self.keep_prob = config.keep_prob
		pass

	def __keep_prob__(self, x):
		if self.keep_prob < 1:
			return tf.nn.dropout(x, keep_prob=self.keep_prob)
		else:
			return x

	def __pooling__(self, x, pooling_way, axis, pos_mask=None):
		if pooling_way == 'max':
			return tf.reduce_max(x, axis=axis)
		if pooling_way == 'avg':
			return tf.reduce_mean(x, axis=axis)
		if pooling_way == 'piece':
			return tf.reduce_max(x * pos_mask, axis=axis)

	def __cnn_cell__(self, x,height,width,num_filters,padding,activation):
		w = tf.get_variable(tf.truncated_normal_initializer(stddev=0.01),
							shape=[height, width, 1, num_filters], name='weight')
		b = tf.get_variable(tf.constant_initializer(0), shape=[num_filters], name='bias')
		conv = tf.nn.conv2d(x, w, strides=[1, 1, width, 1], padding=padding)
		h = activation(tf.nn.bias_add(conv, b), name='h')
		return h

	def cnn(self, x, padding='SAME', filter_sizes=[3,4,5],pooling_way='max', activation=tf.nn.tahn):
		'''
		:param x: [batch_size, config.max_len, width, 1] width要看具体加不加ner等其他信息
		:param padding: ['SAME','VALID']
		:param filter_sizes: default [3，4，5]
		:param pooling_way: ['max, avg']
		:return: [batch_size, len(filter_sizes) * config.num_filters ]
		'''
		width = x.shape[2]
		pooled_outputs = []
		for i,k in enumerate(filter_sizes):
			with tf.variable_scope("conv-%d" % k):
				h = self.__cnn_cell__(x,k,width,config.num_filters,padding,activation)  # [bz, max_len, 1, num_filters]
				pooled = self.__pooling__(h, pooling_way=pooling_way, axis=1) # [bz, 1, 1, num_filters]
				pooled_outputs.append(pooled)
		ret = tf.reshape(tf.concat(pooled_outputs,axis=3),[-1,len(filter_sizes) * config.num_filters])
		return self.__keep_prob__(ret)

	def pcnn(self, x, in_masks, padding='SAME',filter_sizes=[3,4,5], activation=tf.nn.tahn):
		pos_mask = tf.expand_dims(tf.transpose(in_masks, [0, 2, 1]), axis=1)  # [bz,max_len,3,1]
		width = x.shape[2]
		pooled_outputs = []
		for i,k in enumerate(filter_sizes):
			with tf.variable_scope("conv-%d" % k):
				h = self.__cnn_cell__(x,k,width,config.num_filters,padding,activation)  # [bz, max_len, 1, num_filters]
				pooled = self.__pooling__(h, pooling_way='piece', axis=1, pos_mask=pos_mask)  # [bz,1,3,num_filters]
				pooled_outputs.append(tf.concat(pooled,axis=2))
		ret = tf.reshape(tf.concat(pooled_outputs, axis=3), [-1, len(filter_sizes) * config.num_filters * 3])
		return self.__keep_prob__(ret)

	def __rnn_cell__():
		pass

	def rnn():
		pass

	def birnn():
		pass