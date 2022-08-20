#!/usr/bin/env python

from tqdm import tqdm
from annotation import read_dataset, write_dataset
import bs4
import os
import subprocess
import argparse
import pdb


def run_ims(dataset, ims_dir):
    """
    Run IMS (It Makes Sense) to disambiguate words.
    """
    org_dir = os.getcwd()
    os.chdir(ims_dir)
    IMS_IN = 'tmp.txt'
    IMS_OUT = 'tmp.out'
    IMS_CMD = f'java -mx2500m -cp "lib/*" sg.edu.nus.comp.nlp.ims.implement.CTester -ptm lib/tag.bin.gz -tagdir lib/tagdict.txt -prop lib/prop.xml -f sg.edu.nus.comp.nlp.ims.feature.CAllWordsFeatureExtractorCombination -c sg.edu.nus.comp.nlp.ims.corpus.CAllWordsPlainCorpus -r sg.edu.nus.comp.nlp.ims.io.CPlainCorpusResultWriter -is lib/WordNet-3.0/dict/index.sense {IMS_IN} models-MUN-SC-wn30 models-MUN-SC-wn30 {IMS_OUT} -delimeter "/" -split 1 -token 1 -pos 0 -lemma 0'
    FNULL = open(os.devnull, 'w')  # for suppressing IMS output

    # Run IMS per sentence to reduce memory consumption.
    for sent in tqdm([sent for doc in dataset.docs for sent in doc.sents]):
        with open(IMS_IN, 'w') as fout:
            fout.write(f"{' '.join([t.text for t in sent.tokens])}\n")
        subprocess.call(IMS_CMD, stdout=FNULL, stderr=subprocess.STDOUT, shell=True)
        with open(IMS_OUT, 'r') as fin:
            output = fin.read().strip()
        output = bs4.BeautifulSoup(output, 'html.parser')
        # Align IMS output with tokens
        idx = 0
        for item in output:
            if item == ' ':
                continue

            token = sent.tokens[idx]
            if isinstance(item, bs4.element.Tag):
                if token.text.startswith('\n'):
                    continue
                assert item.text == token.text
                assert item.attrs['length'].startswith('1 ')
                best_sense_id = None
                best_prob = 0
                for sense in item.attrs['length'][2:].split(' '):
                    sense_id, prob = sense.split('|')
                    prob = float(prob)
                    if prob > best_prob:
                        best_sense_id = sense_id
                        best_prob = prob
                token.sense_id = best_sense_id
                idx += 1
            else:  # raw strings
                # Hack for token alignments
                if '2 1/2' in item:
                    _tokens = item.strip(' ').replace('2 1/2', '2#1/2').split(' ')
                    _tokens = [t.replace('2#1/2', '2 1/2') for t in _tokens]
                else:
                    _tokens = item.strip(' ').split(' ')

                for _token in _tokens:
                    token = sent.tokens[idx]
                    assert _token.strip('\n') == token.text.strip('\n')  # hack
                    idx += 1

    os.remove(IMS_IN)
    os.remove(IMS_OUT)
    os.chdir(org_dir)
    FNULL.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, metavar='str', default='out/sw100.json.gz',
                        help="input file (gzipped JSON)")
    parser.add_argument('--output', type=str, metavar='str', default='out/sw100_wsd.json.gz',
                        help="output file (gzipped JSON)")
    parser.add_argument('--ims', type=str, metavar='str', default='ims_0.9.2.1',
                        help="IMS directory")
    args = parser.parse_args()

    dataset = read_dataset(args.input)

    print("Running IMS (It Makes Sense) for word sense disambiguation...")
    run_ims(dataset, args.ims)

    print(f"Word sense disambiguration is done.")

    write_dataset(dataset, args.output)
    print(f"Saved output to {args.output}")
