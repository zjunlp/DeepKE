# coding=utf-8
import tensorflow as tf
from framework import Framework
from config import Config
config = Config.config

def cnn(is_training):
	framework = Framework(is_training=True)
	x_emb = framework.x
	pos1_emb = framework.pos1
	pos2_emb = framework.pos2
	d = config.EMBEDDING_DIM + 2 * config.POS_EMBEDDING_DIM
	new_x = tf.reshape(tf.concat([x_emb, pos1_emb, pos2_emb], -1), [-1,config.max_len, d, 1]) # bz, n, d
	h = framework.encoder.cnn(new_x, padding=config.padding, filter_sizes=config.filter_sizes,
							  pooling_way=config.pooling_way, activation=tf.nn.tahn)

	map(h,y)
	if is_training:
		loss =
		out_put =
	else:
		pass

