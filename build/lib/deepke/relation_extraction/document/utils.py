import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn_sample(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    entity_pos = [f["entity_pos"] for f in batch]

    negative_alpha = 8
    positive_alpha = 1
    labels, hts = [], []
    for f in batch:
        randnum = random.randint(0, 1000000)

        pos_hts = f['pos_hts']
        pos_labels = f['pos_labels']
        neg_hts = f['neg_hts']
        neg_labels = f['neg_labels']
        if negative_alpha > 0:
            random.seed(randnum)
            random.shuffle(neg_hts)
            random.seed(randnum)
            random.shuffle(neg_labels)
            lower_bound = int(max(20, len(pos_hts) * negative_alpha))
            hts.append( pos_hts * positive_alpha + neg_hts[:lower_bound] )
            labels.append( pos_labels * positive_alpha + neg_labels[:lower_bound] )

    # labels = [f["labels"] for f in batch]
    # hts = [f["hts"] for f in batch]


    # entity_pos_single = []
    #     # for f in batch:
    #     #     entity_pos_item = f["entity_pos"]
    #     #     entity_pos2 = []
    #     #     for e in entity_pos_item:
    #     #         entity_pos2.append([])
    #     #         mention_num = len(e)
    #     #         bounds = np.random.randint(mention_num, size=3)
    #     #         for bound in bounds:
    #     #             entity_pos2[-1].append(e[bound])
    #     #     entity_pos_single.append( torch.tensor(entity_pos2) )

    output = (input_ids, input_mask, labels, entity_pos, hts, )

    return output


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    entity_pos = [f["entity_pos"] for f in batch]

    labels = [f["labels"] for f in batch]
    hts = [f["hts"] for f in batch]
    output = (input_ids, input_mask, labels, entity_pos, hts )
    return output
