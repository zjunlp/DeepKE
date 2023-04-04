import hashlib
from transformers import XLMRobertaTokenizer, BertTokenizer, RobertaTokenizer

from torch.nn.utils.rnn import pad_sequence


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, "r", encoding="utf-8"):
        value, key = line.strip("\n").split("\t")
        vocab[key] = int(value)
    return vocab

def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w", encoding="utf8") as outfile:
        [outfile.write(d + "\n") for d in data]

def to_crf_pad(org_array, org_mask, pad_label_id):
    crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
    crf_array = pad_sequence(crf_array, batch_first=True, padding_value=pad_label_id)
    crf_pad = (crf_array != pad_label_id)
    # the viterbi decoder function in CRF makes use of multiplicative property of 0, then pads wrong numbers out.
    # Need a*0 = 0 for CRF to work.
    crf_array[~crf_pad] = 0
    return crf_array, crf_pad


def unpad_crf(returned_array, returned_mask, org_array, org_mask):
    out_array = org_array.clone().detach()
    out_array[org_mask] = returned_array[returned_mask]
    return out_array

def label_data(data, start, l, _type):
    """label_data"""
    for i in range(start, start + l):
        suffix = "B-" if i == start else "I-"
        data[i] = "{}{}".format(suffix, _type)
    return data

def process_remained_pred_trigger(trigger_pred):
    raw_labels = ["O"] * len(trigger_pred)
    trigger_labels_list = []
    trigger_label = raw_labels
    for idx, label in enumerate(trigger_pred):
        if label == "O":
            if idx != 0 and trigger_pred[idx-1] != "O":
                trigger_labels_list.append(trigger_label)
            trigger_label = raw_labels
        else:
            trigger_label[idx] = label
    return trigger_labels_list

def clear_wrong_tokens(words, labels, tokenizer, max_len):
    
    tokens_cleaned = []
    labels_cleaned = []
    trigger_start = None

    for idx, word, label in zip(range(len(words)), words, labels):
        token = tokenizer.tokenize(word)
        if label != "O" and trigger_start == None:
            trigger_start = idx
        if len(token) == 0:
            continue
        tokens_cleaned.extend(token)
        labels_cleaned.append(label)

    if len(tokens_cleaned) > max_len:
        tokens_cleaned = tokens_cleaned[:max_len]
        labels_cleaned = labels_cleaned[:max_len]
    
    return tokens_cleaned, labels_cleaned, trigger_start