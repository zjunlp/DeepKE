# coding=utf-8
import tensorflow as tf
from framework import Framework
from config import Config
config = Config.config

def cnn(is_training):
	framework = Framework(is_training=True)
	x = framework.x
	dist1 = framework.dist1
	dist2 = framework.dist2
	ner = framework.ner
	d = config.embedding_dim + 2 * config.dist_embedding_dim + config.ner_embdding_dim
	new_x = tf.reshape(tf.concat([x, dist1, dist2,ner], -1), [-1,config.max_len, d, 1]) # bz, n, d
	if is_training:
		pass
	else:
		pass

