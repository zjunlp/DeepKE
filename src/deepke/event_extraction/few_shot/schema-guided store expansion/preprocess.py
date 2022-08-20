#!/usr/bin/env python

from annotation import write_dataset, Dataset, Document, Sentence, Token, Dependency
import json
import os
import argparse
import pdb


def parse_file(input_file, corenlp_file):
    with open(input_file, 'r') as fin:
        text = fin.read()
    doc = Document()
    doc.doc_id = input_file
    doc.text = text

    with open(corenlp_file, 'r') as fin:
        corenlp_output = json.load(fin)

    # Sentence annotations
    tokenid2token = {}
    for i, sent_dict in enumerate(corenlp_output['sentences']):
        sent = Sentence()
        sent.sent_id = i + 1
        sent.begin = sent_dict['tokens'][0]['characterOffsetBegin']
        sent.end = sent_dict['tokens'][-1]['characterOffsetEnd']
        sent.doc = doc
        doc.sents.append(sent)

        # Token annotations
        tokenid2token.clear()
        for token_dict in sent_dict['tokens']:
            token = Token()
            token.token_id = token_dict['index']
            token.begin = token_dict['characterOffsetBegin']
            token.end = token_dict['characterOffsetEnd']
            token.pos = token_dict['pos']
            token.lemma = token_dict['lemma']
            token.sent = sent
            sent.tokens.append(token)
            if token.text != token_dict['originalText']:
                print("[WARNING] Token misalignment (doc ID: {doc.doc_id}): "
                      "{token.text} vs {token_dict['originalText']}")
            tokenid2token[token.token_id] = token

        # Dependency annotations
        for dep_dict in sent_dict['basicDependencies']:
            dep = Dependency()
            dep.rel = dep_dict['dep']
            tail_token_id = dep_dict['dependent']
            tail_token = tokenid2token[tail_token_id]
            dep.tail = tail_token_id
            tail_token.head_deps.append(dep)

            if dep.rel == 'ROOT':
                tail_token.is_root_dep_node = True
            else:
                tail_token.is_root_dep_node = False
                head_token_id = dep_dict['governor']
                head_token = tokenid2token[head_token_id]
                dep.head = head_token_id
                head_token.tail_deps.append(dep)

    return doc


def parse(input_dir):
    """
    Parse texts to get sentences, tokens, etc.
    """
    dataset = Dataset()
    for dirpath, dirs, files in os.walk(input_dir):
        for f in files:
            txt_file = os.path.join(dirpath, f)
            if not txt_file.endswith('.txt'):
                continue
            if not os.path.exists(txt_file[:-4] + '.ann'):
                continue
            corenlp_file = txt_file + '.json'  # CoreNLP output
            assert os.path.exists(corenlp_file)
            doc = parse_file(txt_file, corenlp_file)
            dataset.docs.append(doc)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, metavar='str', default='data/SW100',
                        help="input directory")
    parser.add_argument('--output', type=str, metavar='str', default='out/sw100.json.gz',
                        help="output file (gzipped JSON)")
    args = parser.parse_args()

    if not os.path.exists('out'):
        os.mkdir('out')

    print("Parsing the dataset...")
    dataset = parse(args.input)

    print(f"Parsing is done.")

    write_dataset(dataset, args.output)
    print(f"Saved output to {args.output}")
