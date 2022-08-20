#!/usr/bin/env python

"""
This code converts annotation in the Brat format.
"""

from annotation import read_dataset
import os
import argparse
import pdb


def convert_to_brat(dataset, output_dir):
    for doc in dataset.docs:
        text_file = os.path.join(output_dir, doc.doc_id)
        ann_file = text_file[:-4] + '.ann'
        output_subdir = os.path.dirname(text_file)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        with open(text_file, 'w') as fout:
            fout.write(doc.text)

        with open(ann_file, 'w') as fout:
            nugget_id = 1
            for sent in doc.sents:
                for nugget in sent.event_nuggets:
                    tokens = [sent.tokens[token_id-1] for token_id in nugget.tokens]
                    offset = create_offset(tokens)
                    fout.write(f"T{nugget_id}\tEvent {offset}\t{nugget.text}\n")
                    fout.write(f"E{nugget_id}\tEvent:T{nugget_id}\n")
                    nugget_id += 1


def create_offset(tokens):
    offsets = []
    for i, token in enumerate(tokens):
        if i > 0 and token.token_id == tokens[i-1].token_id + 1:
            # Next token (continuous)
            offsets[-1][1] = token.end  # update the previous end offset
            continue
        offset = [token.begin, token.end]
        offsets.append(offset)

    return ';'.join([f'{begin} {end}' for begin, end in offsets])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, metavar='str', default='out/sw100_wsd_phrase_rule.json.gz',
                        help="input file (gzipped JSON)")
    parser.add_argument('--output', type=str, metavar='str', default='out',
                        help="output directory")
    args = parser.parse_args()

    dataset = read_dataset(args.input)
    print(f"Loaded input data ({len(dataset.docs)} documents) from {args.input}")

    convert_to_brat(dataset, args.output)
    print(f"Converted annotation in the Brat format under '{args.output}' directory.")
