import os
import numpy as np
import json
import shutil




def get_labels(path, name,  negative_label="no_relation"):
    """See base class."""

    with open(path + "/" + name, "r") as f:
        features = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                features.append(eval(line))
    return features

def generate_k_shot(data_dir):
    Seed = [1, 2, 3, 4, 5]
    mode = 'k-shot'
    data_file = 'train.txt'
    path = 'data'

    output_dir = os.path.join(path, mode)
    dataset = get_labels(path, data_file)

    for seed in Seed:

        # Other datasets
        np.random.seed(seed)
        np.random.shuffle(dataset)

        # Set up dir
        k = 8
        setting_dir = os.path.join(output_dir, f"{k}-{seed}")
        os.makedirs(setting_dir, exist_ok=True)

        label_list = {}
        for line in dataset:
            label = line['relation']
            if label not in label_list:
                label_list[label] = [line]
            else:
                label_list[label].append(line)

        with open(os.path.join(setting_dir, "train.txt"), "w") as f:
            file_list = []
            for label in label_list:
                for line in label_list[label][:k]:  # train中每一类取前k个数据
                    f.writelines(json.dumps(line))
                    f.write('\n')

            f.close()

    shutil.copyfile('data/rel2id.json','data/k-shot/8-1/rel2id.json')
    shutil.copyfile('data/val.txt','data/k-shot/8-1/val.txt')
    shutil.copyfile('data/test.txt','data/k-shot/8-1/test.txt')
