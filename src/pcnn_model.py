#coding: utf-8
import tensorflow as tf
import config


class Model(object):
	def __init__(self, embeddings, batch_size, is_training=True,):
		bz = config.BATCH_SIZE
		dw = config.EMBEDDING_DIM
		dp = config.POS_EMBEDDING_DIM
		np = config.POS_EMBEDDING_NUM
		n = config.MAX_LEN
		# k = config.slide_window
		dc = config.NUM_FILTERS
		# nr = config.CLASS_NUM # number of relations
		nr = 2 # number of relations
		keep_prob = config.KEEP_PROB

		# input
		in_bz = tf.placeholder(dtype=tf.int32, shape=[], name='in_bz')
		in_x = tf.placeholder(dtype=tf.int32, shape=[None,n], name='in_x') # sentences
		in_e1 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='in_e1')
		in_e2 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='in_e2')
		in_rel = tf.placeholder(dtype=tf.int32, shape=[None], name='in_rel')
		in_masks = tf.placeholder(dtype=tf.float32, shape=[None, 3, n], name='in_masks')
		in_dist1 = tf.placeholder(dtype=tf.int32, shape=[None,n], name='in_dist1')
		in_dist2 = tf.placeholder(dtype=tf.int32, shape=[None,n], name='in_dist2')
		# in_pos = tf.placeholder(dtype=tf.int32, shape=[None,n], name='in_pos')
		in_ner = tf.placeholder(dtype=tf.int32, shape=[None,n], name='in_ner')	
		in_y = tf.placeholder(dtype=tf.int32, shape=[None], name='in_y') 		

		# self.inputs = (in_bz, in_masks, in_x, in_e1, in_e2, in_dist1, in_dist2,in_ner, in_y)
		self.inputs = (in_bz, in_masks, in_x, in_e1, in_e2, in_rel,in_dist1, in_dist2,in_ner, in_y)
		# self.inputs = (in_bz, in_masks, in_x, in_e1, in_e2, in_dist1, in_dist2, in_pos, in_ner, in_y)
		
		# embeddings
		initializer = tf.truncated_normal_initializer(stddev=0.01)
		bia_initializer = tf.constant_initializer(0)
		pos_embed_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='word_embed')
		pos1_embed = tf.get_variable(initializer=pos_embed_initializer,shape=[np, dp],name='position1_embed')
		pos2_embed = tf.get_variable(initializer=pos_embed_initializer,shape=[np, dp],name='position2_embed')
		pos1_embed = tf.nn.l2_normalize(pos1_embed, dim=1)
		pos2_embed = tf.nn.l2_normalize(pos2_embed, dim=1)
		# pos_embed = tf.get_variable(initializer=initializer,shape=[np, 10],name='pos_embed')
		ner_embed = tf.get_variable(initializer=initializer,shape=[np, 10],name='ner_embed')
		rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc],name='relation_embed')

		# embdding lookup
		in_rel = tf.reshape(in_rel,[-1,1])
		e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,3,dw
		e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,3,dw
		rel = tf.nn.embedding_lookup(embed, in_rel, name='rel')
		x = tf.nn.embedding_lookup(embed, in_x, name='x')   # bz,n,dw
		dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')#bz, n, k,dp
		dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')# bz, n, k,dp
		# pos = tf.nn.embedding_lookup(pos_embed, in_pos, name='pos')# bz, n, dp
		ner = tf.nn.embedding_lookup(ner_embed, in_ner, name='ner')# bz, n, dp
		y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc
		# new_x = x
		new_x = new_x * rel

		# convolution
		# x: (batch_size, max_len, embdding_size, 1)
		# w: (filter_size, embdding_size, 1, num_filters)
		d = dw+2*dp + 10
		# d = dw+2*dp + 10 + 10   # consider pos
		#filter_sizes = [3, 4, 5]
		filter_sizes = [3, 4, 5]
		pooled_outputs = []

		# x_conv = tf.reshape(tf.concat([new_x, dist1, dist2,pos,ner], -1), # bz, n, d
		# 						[-1,n,d,1])

		x_conv = tf.reshape(tf.concat([new_x, dist1, dist2,ner], -1), # bz, n, d
								[-1,n,d,1])

		if is_training and keep_prob < 1:
			x_conv = tf.nn.dropout(x_conv, keep_prob)

		pos_mask = tf.expand_dims(tf.transpose(in_masks, [0, 2, 1]), axis=1) # bz, 1, n, 3

		filter_sizes = [3,4,5]
		pooled_outputs = []


		for i,k in enumerate(filter_sizes):
			with tf.variable_scope("conv-%d" % k):
				w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight')
				b = tf.get_variable(initializer=bia_initializer,shape=[dc],name='bias')
				conv = tf.nn.conv2d(x_conv, w, strides=[1,1,d,1],padding="SAME")

				h = tf.squeeze(tf.nn.tanh(tf.nn.bias_add(conv,b),name="h")) # bz, n, 1, dc -> bz, n, dc
				h = tf.expand_dims(tf.transpose(h, [0,2,1]), axis=-1) # bz, dc, n, 1
				pcnn_pool = tf.reduce_max(h * pos_mask, axis=2) #bz, dc, n, 3 -> bz, dc, 3
				# print(pcnn_pool.get_shape())

				#pooled_outputs.append(tf.concat([pooled_1, pooled_2, pooled_3], -1))

			#h_pool = tf.concat(pooled_outputs, 0)
		h_pool_flat = tf.reshape(pcnn_pool, [-1,dc*3]) #bz, dc * 3
	

		# # e embdding
		# # e1_sum =  tf.reduce_sum(e1,axis=1)
		# # e2_sum =  tf.reduce_sum(e2,axis=1),
		# e_flat = tf.concat([e1, e2], 2)
		# e_flat = tf.reshape(e_flat,[-1,dw*2*3])
		# all_flat = tf.concat([h_pool_flat, e_flat],1)
		# h_pool_flat = tf.reshape(all_flat,[-1,dc*len(filter_sizes) + dw * 2*3])
		all_flat = h_pool_flat
		h_pool_flat = tf.reshape(all_flat,[-1,dc*len(filter_sizes)])

		if is_training and keep_prob < 1:
			h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)
		
		
		# output
		# W1 = tf.get_variable(initializer=initializer,shape=[dc*len(filter_sizes)  + dw * 2*3, 100],name='w1')
		W1 = tf.get_variable(initializer=initializer,shape=[dc*len(filter_sizes), 100],name='w1')
		b1 = tf.get_variable(initializer=bia_initializer,shape=[100],name='b1')
		o1 = tf.nn.xw_plus_b(h_pool_flat,W1,b1,name="o1")

		W2 = tf.get_variable(initializer=initializer,shape=[100, 100],name='w2')
		b2 = tf.get_variable(initializer=bia_initializer,shape=[100],name='b2')
		o2 = tf.nn.xw_plus_b(o1,W2,b2,name="o2")

		# W_o = tf.get_variable(initializer=initializer,shape=[100, dw],name='w_o')
		# b_o = tf.get_variable(initializer=bia_initializer,shape=[dw],name='b_o')
		W_o = tf.get_variable(initializer=initializer,shape=[100,nr],name='w_o')
		b_o = tf.get_variable(initializer=initializer,shape=[nr],name='b_o')
		scores = tf.nn.xw_plus_b(o2,W_o,b_o,name="scores")
		self.prob = scores
		predict = tf.argmax(scores,axis=1,name="predictions") # 返回最大值所在的下标,默认是int64
		predict = tf.cast(predict, dtype=tf.int32)
		self.preds = predict
		correct_num = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32)) # equal 返回的是[True,False]，cast把它映射到int32[1,0]
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, in_y), dtype=tf.float32)) # accuracy = correct_num / total_num
		# self.predict = predict
		self.acc = correct_num

		loss = tf.reduce_mean(
		  #tf.nn.softmax_cross_entropy_with_logits(logits=scores, 
		  #                                        labels=tf.one_hot(in_y, nr))
		  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=in_y)
		)
		l2_loss = tf.nn.l2_loss(W_o)
		l2_loss += tf.nn.l2_loss(b_o)
		l2_loss = config.L2_REG_LAMBDA * l2_loss
		
		self.loss = loss + l2_loss

		if not is_training:
			return

		# optimizer 
		# optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
		optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE)
		# optimizer2 = tf.train.AdamOptimizer(config.learning_rate2)

		# tvars = tf.trainable_variables()
		# grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
		#                                   config.grad_clipping)
		# capped_gvs = zip(grads, tvars)

		# tf.logging.set_verbosity(tf.logging.ERROR)
		global_step = tf.Variable(0, trainable=False, name='global_step')
		# train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
		# reg_op = optimizer2.minimize(l2_loss)

		self.train_op = optimizer.minimize(self.loss)
		self.reg_op = tf.no_op()
		self.global_step = global_step