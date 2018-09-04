# coding = utf-8

class Vocab(object):
	"""docstring for Vocab"""
	def __init__(self, filename=None, initial_tokens=None, lower=False):
		super(Vocab, self).__init__()
        self.id2token = {}
        self.token2id = {}
        # ner需要再有一个 self.ner2id()
        # self.id2ner = {} 
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_from_file(filename)

    def get_id(self,token):
    	pass

    def get_token(self,id):
    	pass

    def load_from_file(self,file_path):
    	'''
    	使用分段的file或者直接只是token的file，可能需要确定下
    	'''
    	pass

    def add(self,token):
    	if token in self.token2id.keys():
    		return
    	else:
    		idx = len(self.id2token)
    		self.token2id[token] = idx
    		self.id2token[idx] = token

    def randomly_init_embeddings(self, embed_dim):
    	# 随机初始化embedding，<ukw> <blank> 设置为0
    	# 为self.embeddings赋值
    	pass
    
    def load_pretrained_embeddings(self, embedding_path):
    	# 加载预训练的embedding
    	# 为self.embeddings赋值
    	pass

    def convert_to_ids(self, tokens):
    	# 将一句segment_sentence的token转为相应的id
    	pass
    
