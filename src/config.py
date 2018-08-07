# coding=utf-8
ROOT_DIR = "../"
DATA_DIR = ROOT_DIR + "data/"
MODEL_DIR = ROOT_DIR+'model/'
RESULT_DIR = ROOT_DIR+'result/'
EMBEDDING_DIR = DATA_DIR+'embedding/'
PREDICT_FILE_NAME = 'predict.txt'  # 将预测的文件名

# 常需要改的参数
GPU_ID = '1' 	# 使用哪一个gpu
which_model = 'pcnn'
train_ratio = 0.75 # 训练和测试的比例
specific = None # 默认在全部类上分类训练，如果只训练2，3类，设置为specific = [2,3,]

EMBEDDING_DIM = 200
L2_REG_LAMBDA = 0
HIDDEN_DIM = 300
MAX_LEN = 300 # 最终输入到模型中句子长度，单词个数
MAX_CHARACTER_LEN = 600 # 初始过滤限制的句子长度，字的个数
MIN_CHARACTER_LEN = 12
POS_EMBEDDING_DIM = 5
POS_EMBEDDING_NUM = 2 * MAX_LEN
NUM_FILTERS = 200
LEARNING_RATE = 0.004
KEEP_PROB = 0.5
EPOCHES = 20
BATCH_SIZE = 50
EXTRA_DICT_PATH = DATA_DIR + 'customed_dict.txt' # 分词时用户自定义词典的路径

which_relation = 'per' # 高管：'per'；公司之间:'com'
per_com_relation = ['法定代表人','董事长','董事','总经理','总裁','副总经理','副总裁','财务总监','监事','独立董事',]
CLASS_NUM = len(per_com_relation) + 1 # 注意有第 0 类
TRAIN_FILE = DATA_DIR+ which_relation + '_multi_train.csv'
TEST_FILE = DATA_DIR + which_relation +'_multi_test.csv'
TRAIN_SAVE_PATH = RESULT_DIR + which_relation + '_train_' + which_model + '_' + str(CLASS_NUM)
PREDICT_SAVE_PATH = RESULT_DIR + which_relation + '_predict_' + which_model + '_' + str(CLASS_NUM)
MODEL_PATH = MODEL_DIR + which_relation + '_' + which_model + '_' + str(CLASS_NUM)