import os
import json
import csv
from tqdm import *
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ds_label_data')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'cn'])
    parser.add_argument('--source_file', type=str, default='source_data.json')
    parser.add_argument('--triple_file', type=str, default='triple_file.csv')
    parser.add_argument('--train_rate', type=float, default=0.8)
    parser.add_argument('--dev_rate', type=float, default=0.1)
    parser.add_argument('--test_rate', type=float, default=0.1)
    args = parser.parse_args()
    print(args)
    
    dic = csv.reader(open(os.path.join(os.getcwd(), args.triple_file), 'r', encoding='utf-8'))  # read dict
    dic = list(dic)[1:]  # remove the header

    source = json.load(open(os.path.join(os.getcwd(), args.source_file), 'r', encoding='utf-8'))

    labeled_data = []
    with tqdm(desc='labeling data', total=len(source)) as pbar:
        for src_data in source:
            pbar.update(10)
            data = {}
            sentence = src_data['sentence']
            head = src_data['head']
            tail = src_data['tail']
            data['sentence'] = sentence
            data['head'] = head
            data['tail'] = tail
            if args.language == 'en':
                head, tail = head.lower(), tail.lower()
            for triple in dic:
                h, t = triple[0], triple[1]
                if args.language == 'en':
                    h, t = h.lower(), t.lower()
                if h == head and t == tail:  # string full match
                    data['relation'] = triple[2]
                    break
            if 'relation' not in data:  # no match
                data['relation'] = 'None'
            labeled_data.append(data)

    # split the dataset
    assert args.train_rate + args.dev_rate + args.test_rate == 1.0
    total = len(labeled_data)
    train_len = int(total * args.train_rate)
    dev_len = int(total * args.dev_rate)
    test_len = int(total * args.test_rate)
    train_data = labeled_data[:train_len]
    dev_data = labeled_data[train_len:train_len+dev_len]
    test_data = labeled_data[-test_len:]

    # write
    json.dump(train_data, open('labeled_train.json', 'w', encoding='utf-8'))
    json.dump(dev_data, open('labeled_dev.json', 'w', encoding='utf-8'))
    json.dump(test_data, open('labeled_test.json', 'w', encoding='utf-8'))
