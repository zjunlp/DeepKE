class TrainingConfig(object):
    seed = 1
    use_gpu = True
    gpu_id = 0
    epoch = 30
    learning_rate = 1e-3
    decay_rate = 0.5
    decay_patience = 3
    batch_size = 64
    train_log = True
    log_interval = 10
    show_plot = True
    f1_norm = ['macro', 'micro']


class ModelConfig(object):
    word_dim = 50
    pos_size = 102  # 2 * pos_limit + 2
    pos_dim = 5
    feature_dim = 60  # 50 + 5 * 2
    hidden_dim = 100
    dropout = 0.3


class CNNConfig(object):
    use_pcnn = True
    out_channels = 100
    kernel_size = [3, 5, 7]


class RNNConfig(object):
    lstm_layers = 3
    last_hn = False


class GCNConfig(object):
    # TODO
    pass


class TransformerConfig(object):
    transformer_layers = 3


class CapsuleConfig(object):
    num_primary_units = 8
    num_output_units = 10  # relation_type
    primary_channels = 1
    primary_unit_size = 768
    output_unit_size = 128
    num_iterations = 3


class LMConfig(object):
    # lm_name = 'bert-base-chinese'  # download usage
    # cache file usage
    lm_file = 'bert_pretrained'
    # transformer 层数，初始 base bert 为12层
    # 但是数据量较小时调低些反而收敛更快效果更好
    num_hidden_layers = 2

class Config(object):
    # 原始数据存放位置
    data_path = 'data/origin'
    # 预处理后存放文件的位置
    out_path = 'data/out'

    # 是否将句子中实体替换为实体类型
    replace_entity_by_type = True
    # 是否为中文数据
    is_chinese = True
    # 是否需要分词操作
    word_segment = True

    # 关系种类
    relation_type = 10

    # vocab 构建时最低词频控制
    min_freq = 2

    # position limit
    pos_limit = 50  # [-50, 50]

    # (CNN, RNN, GCN, Transformer, Capsule, LM)
    model_name = 'Capsule'

    training = TrainingConfig()
    model = ModelConfig()
    cnn = CNNConfig()
    rnn = RNNConfig()
    gcn = GCNConfig()
    transformer = TransformerConfig()
    capsule = CapsuleConfig()
    lm = LMConfig()


config = Config()
