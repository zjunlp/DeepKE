from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
import json

def split_label_words(tokenizer, label_list):
    label_word_list = []
    for label in label_list:
        if label == 'no_relation':
            label_word_id = tokenizer.encode('None', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            tmps = label
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list


def get_label_word(cfg):
    model_name_or_path = cfg.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with open("data/rel2id.json", "r") as file:
        t = json.load(file)
        label_list = list(t)

    t = split_label_words(tokenizer, label_list)

    with open(f"data/{model_name_or_path}.pt", "wb") as file:
        torch.save(t, file)

