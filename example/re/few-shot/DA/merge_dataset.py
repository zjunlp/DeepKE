import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_files', '-i', nargs='+',
                    help='A list of input files containing datasets to merge')
parser.add_argument('--output_file', '-o',
                    help='Output file containing merged dataset')

args = parser.parse_args()

reduced_samples = 0
triples = set()
all_samples = []
for fname in tqdm(args.input_files,desc="Merge"):
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            sentence = ' '.join(line['token'])
            triple = (sentence, line['h']['name'],line['t']['name'])
            if triple in triples:
                reduced_samples += 1
            else:
                triples.add(triple)
                all_samples.append(line)
print('Reduced {} repeated samples'.format(reduced_samples))
with open(args.output_file, 'w') as f:
    for line in all_samples:
        f.writelines(json.dumps(line,ensure_ascii=False))
        f.write('\n')
