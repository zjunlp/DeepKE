# 原始文件位置

class Config(object):
    data_path = 'data/origin'
    # 预处理后存放文件的位置
    out_path = 'data/out'

    # 是否为中文数据
    is_chinese = True
    # 是否需要分词操作
    word_segment = True

    # 关系种类
    relation_type = 10

    # vocab 构建时最低词频控制
    min_freq = 2

    # position embedding
    pos_limit = 50  # [-50, 50]
    pos_size = 102  # 2 * pos_limit + 2

    # model name
    # (CNN, BiLSTM, Transformer, Capsule, Bert)
    model_name = 'CNN'

    # model
    word_dim = 50
    pos_dim = 5

    # feature_dim = 50 + 5 * 2
    hidden_dim = 100
    dropout = 0.3

    # PCNN config
    use_pcnn = True
    out_channels = 100
    kernel_size = [3, 5]

    # BiLSTM
    lstm_layers = 2
    last_hn = False

    # Transformer
    transformer_layers = 2

    # Capsule
    num_primary_units=8
    num_output_units=10    # relation_type
    primary_channels=1
    primary_unit_size=768
    output_unit_size=128
    num_iterations=5

    # Bert
    lm_name = 'bert-base-chinese'

    # train
    seed = 1
    use_gpu = True
    gpu_id = 3
    epoch = 30
    learning_rate = 1e-3
    decay_rate = 0.5
    decay_patience = 3
    batch_size = 64
    train_log = True
    log_interval = 10
    show_plot = True
    f1_norm = ['macro', 'micro']




def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)


        print('*************************************************')
        print('user config:')
        for k, v in kwargs.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


Config.parse = parse

config =Config()
