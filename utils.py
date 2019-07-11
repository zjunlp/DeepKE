import os
import json
import torch
import random
import numpy as np
from datetime import datetime





def random_split(data):
    """
    load random order file to split the dataset
    otherwise generate a new one
    """
    file_name = "data/random_split.json"
    if not os.path.exists(file_name):
        # generate a new order and save to disk
        order = list(range(len(data)))
        np.random.shuffle(order)
        json.dump(order, open(file_name, 'w'), indent=4)
    else:
        # load pre-defined order
        print('Loading a pre-defined order file')
        order = json.load(open(file_name))
    return order


def pad(sent, max_len):
    """
    syntax "[0] * int" only works properly for Python 3.5+
    Note that in testing time, the length of a sentence
    might exceed the pre-defined max_len (of training data).
    """
    l = len(sent)
    if l < max_len:
        return (sent + [0] * (max_len - l))[:max_len]
    else:
        return sent[:max_len]


def snapshot(model, epoch, save_path='snapshot'):
    """
    Saving model w/ its params.
        Get rid of the ONNX Protocal.
    F-string feature new in Python 3.6+ is used.
    """
    os.makedirs(save_path, exist_ok=True)
    current = datetime.now()
    timestamp = f'{current.month:02d}{current.day:02d}_{current.hour:02d}{current.minute:02d}'
    torch.save(
        model.state_dict(),
        save_path + f'/{type(model).__name__}_{timestamp}_{epoch}th_epoch.pkl')
