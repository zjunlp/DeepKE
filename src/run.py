# coding = utf-8
from config import Config
FLAGS = Config.config

def prepare():
	# 这里进行label,训练embedding
	data = Dataset()
	vocab = Vocab()


def train(data,vocab):

	pass

def predict():
	pass


def run():
    logger.info('Running with args : {}'.format(FLAGS))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
	if FLAGS.prepare:
        prepare()
    if FLAGS.train:
        train()
    if FLAGS.predict:
        predict()

if __name__ == '__main__':
    run()





