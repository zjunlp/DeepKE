# coding=utf-8
import random
import torch
import numpy as np

class Config(object):
    # reproducibility
    seed = 4

    #data
    data_path = 'data/origin'

    # vocab
    min_freq = 2

    # model hyterparams
    embedding_size = 128
    hidden_size = 150
    output_size = 6
    num_layers = 1


    # train iter
    epochs = 20
    batch_size = 32
    lr = 5e-4

    # gpu
    gpu = 0

    # save path
    save_step = 1
    save_path = 'snapshot'


config = Config()


def set_seed(seed):
    """
    Freeze every seed.
    All about reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)