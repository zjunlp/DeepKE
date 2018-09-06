# coding=utf-8
import tensorflow as tf
from layer.encoder import Encoder
from config import Config
config = Config.config
from dataset import Dataset

class Framework(object):
	"""docstring for Model"""
	def __init__(self,data,vocab):
		self.initializer = tf.truncated_normal_initializer(stddev=0.01)
		self.bia_initializer = tf.constant_initializer(0)
		self.setup_placeholders()
		self.embedding_lookup(vocab.embeddings)
		self.encoder = Encoder()
		# session info
		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=sess_config)

	def build_graph(self,vocab):
		self._encode()
		self._compute_loss()
		self._create_train_op()

	def setup_placeholders(self,):
		'''
		placeholder : ner,mask
		'''
		self.in_x = tf.placeholder(dtype=tf.int32, shape = [None, config.max_len],name = 'in_sent')
		self.in_e1 = tf.placeholder(dtype=tf.int32, shape = [None,],name = 'in_e1')
		self.in_e2 = tf.placeholder(dtype=tf.int32, shape = [None,],name = 'in_e2')
		self.in_mask = tf.placeholder(dtype=tf.int32, shape = [None,3,config.max_len],name = 'in_mask')
		self.in_pos1 = tf.placeholder(dtype=tf.int32, shape = [None, config.max_len],name = 'in_pos1')
		self.in_pos2 = tf.placeholder(dtype=tf.int32, shape = [None, config.max_len],name = 'in_pos2')
		self.in_ner = tf.placeholder(dtype=tf.int32, shape = [None, config.max_len],name = 'in_ner')
		if config.multi_label:
			self.in_y = tf.placeholder(dtype=tf.int32, shape = [None,config.class_num], name='in_y')
		else:
			self.in_rel = tf.placeholder(dtype=tf.int32, shape = [None,], name='in_rel')
			self.in_y = tf.placeholder(dtype=tf.int32, shape = [None,],name = 'in_y')
		pass

	def embedding_lookup(self,embeddings):
		dist_embed_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='word_embed')
		dist1_embed = tf.get_variable(initializer=dist_embed_initializer, shape=[config.pos_embedding_num, config.pos_embedding_dim], name='position1_embed')
		dist2_embed = tf.get_variable(initializer=dist_embed_initializer, shape=[config.pos_embedding_num, config.pos_embedding_dim], name='position2_embed')
		ner_embed = tf.get_variable(initializer=self.initializer, shape=[config.max_len,config.ner_embedding_dim], name='ner_embed')
		# rel_embed = tf.get_variable(initializer=self.initializer, shape=[nr, dc], name='relation_embed')

		self.x = tf.nn.embedding_lookup(embed, self.in_x, name='x')  # bz,n,dw
		self.e1 = tf.nn.embedding_lookup(embed, self.in_e1, name='e1')  # bz,3,dw
		self.e2 = tf.nn.embedding_lookup(embed, self.in_e2, name='e2')  # bz,3,dw
		rel = tf.nn.embedding_lookup(embed, self.in_rel, name='rel')
		self.dist1 = tf.nn.embedding_lookup(dist1_embed, self.in_pos1, name='dist1')  # bz, n, k,dp
		self.dist2 = tf.nn.embedding_lookup(dist2_embed, self.in_pos2, name='dist2')  # bz, n, k,dp
		self.ner = tf.nn.embedding_lookup(ner_embed, self.in_ner, name='ner')  # bz, n, dp
		self.y = self.in_y

		if config.multi_label is not True:
			self.rel = tf.nn.embedding_lookup(embed, self.in_rel, name='x')  # bz,n,dw

	def classfier(self,x,in_dim,out_dim):
		'''
		隐层到输出的过程
		'''
		# output
		W1 = tf.get_variable(initializer=self.initializer, shape=[in_dim, 100], name='w1')
		b1 = tf.get_variable(initializer=self.bia_initializer, shape=[100], name='b1')
		o1 = tf.nn.xw_plus_b(x, W1, b1, name="o1")

		W2 = tf.get_variable(initializer=self.initializer, shape=[100, 100], name='w2')
		b2 = tf.get_variable(initializer=self.bia_initializer, shape=[100], name='b2')
		o2 = tf.nn.xw_plus_b(o1, W2, b2, name="o2")

		# W_o = tf.get_variable(initializer=self.initializer,shape=[100, dw],name='w_o')
		# b_o = tf.get_variable(initializer=self.bia_initializer,shape=[dw],name='b_o')
		W_o = tf.get_variable(initializer=self.initializer, shape=[100, out_dim], name='w_o')
		b_o = tf.get_variable(initializer=self.initializer, shape=[out_dim], name='b_o')
		scores = tf.nn.xw_plus_b(o2, W_o, b_o, name="scores")
		self.W_o = W_o
		self.b_o = b_o
		self.prob = scores
		predict = tf.argmax(scores, axis=1, name="predictions")  # 返回最大值所在的下标,默认是int64
		predict = tf.cast(predict, dtype=tf.int32)
		self.preds = predict

	def compute_loss(self,):
		loss = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prob, labels=self.in_y)
		)
		l2_loss = tf.nn.l2_loss(self.W_o)
		l2_loss += tf.nn.l2_loss(self.b_o)
		l2_loss = config.L2_REG_LAMBDA * l2_loss
		self.loss = loss + l2_loss

	def create_train_op(self,learning_rate,Optimizer = tf.train.AdamOptimizer):
		global_steps = tf.Variable(0, trainable=False,name='global_step')
		self.global_steps = global_steps
		optimizer = Optimizer(learning_rate)
		self.train_op = optimizer.minimize(self.loss,global_step = self.global_steps)
		self.reg_op = tf.no_op()

	def feed_dict(self,data):
		feed_dict = {}
		feed_dict['self.in_x'] = data.sent
		feed_dict['self.in_mask'] = data.mask
		# ...
		self.feed_dict = feed_dict
		pass

	def train(self,):
		dataset_train = Dataset(config.data_path + config.train_file)
		dataset_test = Dataset(config.data_path + config.test_file)
		test_seq = dataset_test.get_data()
		train_iter = dataset_train.one_mini_batch(config.batch_size,shuffle=True)
		# 如何构建多分类
		pass


	def predict(self,):
		dataset_predict = Dataset(config.data_path + config.predict_file)
		predict_data = dataset_predict.get_data()

		pass

	def save(self,):
		pass

	def restore(self,):
		pass
