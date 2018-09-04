# coding=utf-8

class DottableDict(dict):
	# 这样dict能通过点访问如 config.max_len
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()


class Config(object):
	'''
	初始的参数，可以写入这里，
	'''
	config = DottableDict()
	config.allowDotting()
	config['per_com_relation'] = ['法定代表人','董事长','董事','总经理','总裁','副总经理','副总裁','财务总监','监事','独立董事',]
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


	# 其余的也放在其中
