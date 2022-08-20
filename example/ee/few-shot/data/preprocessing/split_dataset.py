import argparse
import json
from collections import Counter, defaultdict
import os
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", type=str, required=True)
parser.add_argument("-s", "--split_path", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True)
args = parser.parse_args()

inp = [json.loads(line) for line in open(args.input_path, 'r')]
split = [x.strip('\n') for x in open(args.split_path, 'r')]

counter = 0
with open(args.output_path, 'w') as f:
    for doc in inp:
        if doc['doc_id'] in split:
            counter += 1
            f.write(json.dumps(doc) + '\n')

print('Processed {} number of instances'.format(counter))