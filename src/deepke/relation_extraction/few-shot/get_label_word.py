from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json
import argparse




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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    # parser.add_argument("--task", type=str, nargs="+",
    #     default=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'],
    #     help="Task names")
    parser.add_argument("--seed", type=int, nargs="+",
        default=[1, 2, 3],
        help="Random seeds")

    parser.add_argument("--model_name_or_path", type=str, default="bert-large-uncased")
    parser.add_argument("--dataset_name", type=str, default="semeval")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    dataset_name = args.dataset_name

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with open(f"dataset/{dataset_name}/rel2id.json", "r") as file:
        t = json.load(file)
        label_list = list(t)

    t = split_label_words(tokenizer, label_list)

    with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
        torch.save(t, file)

if __name__ == "__main__":
    main()

