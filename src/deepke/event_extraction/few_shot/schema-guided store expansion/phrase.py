#!/usr/bin/env python

"""
A simple phrase detector.  This algorithm is based up dictionary-lookups with
dependency parse.
"""

from annotation import read_dataset, write_dataset, get_head_word_index, Phrase
import gzip
import argparse
import pdb


PHRASE_END = '</E>'


def detect_phrases(dataset, phrase_dict, deps):
    """
    Detects phrases.
    """
    for doc in dataset.docs:
        for sent in doc.sents:
            detect_phrases_in_sentence(sent, phrase_dict, deps)


def detect_phrases_in_sentence(sent, phrase_dict, deps):
    phrase_id = 1

    # We might want to skip a part of already found discontinuous phrase,
    # e.g., on in 'turn A on'
    skip_token_idxs = {}
    num_tokens = len(sent.tokens)
    token2idx = {t:idx for idx, t in enumerate(sent.tokens)}
    p = None  # pointer to a phrase dictionary
    i = 0
    while i < num_tokens:
        if i in skip_token_idxs:
            i += 1
            continue

        token = sent.tokens[i]
        lemma = None if token.lemma is None else token.lemma.lower()
        word = token.text.lower()
        if lemma not in phrase_dict and word not in phrase_dict:
            # This word does not possibly form a phrase
            p = None
            i += 1
            continue

        # At this point, a word is possibly a part of a phrase.
        if lemma in phrase_dict:
            p = phrase_dict[lemma]  # pointer
        else:
            p = phrase_dict[word]
        phrase_offsets = []
        for offset in range(1, num_tokens-i):  # seeking phrases
            if offset > 1 and PHRASE_END in p:  # found a phrase
                phrase_offsets.append(offset)

            idx = i + offset
            if idx in skip_token_idxs:
                continue
            next_token = sent.tokens[idx]
            next_lemma = None if next_token.lemma is None else next_token.lemma.lower()
            next_word = next_token.text.lower()
            if next_lemma not in p and next_word not in p:
                break  # stop seeking phrases
            if next_lemma in p:
                p = p[next_lemma]
            else:
                p = p[next_word]

        next_offset = 1
        if phrase_offsets:  # found continuous phrases
            for po in reversed(phrase_offsets):
                token_idxs = [idx for idx in range(i, i+po)]
                if check_dependency_closure(sent, token_idxs):
                    # precedence over longer phrases
                    create_phrase(sent, token_idxs, phrase_id)
                    phrase_id += 1
                    next_offset = po
                    break
        else:
            # Seek discontinuous phrases if the current token is a verb.
            # Here, we assume that discontinuous verbal pharses consist of
            # two words, not more.
            if lemma in phrase_dict:
                p = phrase_dict[lemma]  # pointer
            else:
                p = phrase_dict[word]
            if token.pos.startswith('VB'):
                for tail_dep in token.tail_deps:
                    if not any([dep for dep in deps if tail_dep.rel.startswith(dep)]):
                        continue
                    tail = tail_dep.tail
                    if tail < token.token_id:  # the tail precedes the current token
                        continue
                    tail = token.sent.tokens[tail-1]
                    tail_lemma = None if tail.lemma is None else tail.lemma.lower()
                    if tail_lemma in p and PHRASE_END in p[tail_lemma]:
                        tail_idx = token2idx[tail]
                        create_phrase(sent, [i, tail_idx], phrase_id)
                        phrase_id += 1
                        skip_token_idxs[tail_idx] = True
                        # debug
                        # print(sent.doc.doc_id, sent.text)
                        # print("{} -- {} --> {}".format(token.text, tail_dep.rel, tail_dep.tail.text))
                        break
                    tail_word = tail.text.lower()
                    if tail_word in p and PHRASE_END in p[tail_word]:
                        tail_idx = token2idx[tail]
                        create_phrase(sent, [i, tail_idx], phrase_id)
                        phrase_id += 1
                        skip_token_idxs[tail_idx] = True
                        # debug
                        # print(sent.doc.doc_id, sent.text)
                        # print("{} -- {} --> {}".format(token.text, tail_dep.rel, tail_dep.tail.text))
                        break

        i += next_offset


def check_dependency_closure(sent, token_idxs):
    """
    Returns true if the given token (continuous or discontinuous) spans form
    dependency closure.
    """
    tokens = [sent.tokens[idx] for idx in token_idxs]
    head_word_idx = get_head_word_index(tokens)
    if head_word_idx is None:
        return False
    head_word = tokens[head_word_idx]
    token2idx = {t : True for t in tokens}
    for i, token in enumerate(tokens):
        if i == head_word_idx:
            continue
        if not check_head_nodes(token, token2idx, head_word):
            return False
    return True


def check_head_nodes(token, token2idx, head_word):
    loop, max_loop = 0, 20
    curr_token = token
    while True:
        if loop > max_loop:
            break
        if not curr_token.head_deps:
            break
        h = curr_token.head_deps[0].head
        if h not in token2idx:
            break
        if h == head_word:
            return True
        curr_token = h
        loop += 1
    return False


def create_phrase(sent, token_idxs, phrase_id):
    """
    Creates a phrase with the given token (continuous or discontinuous) spans.
    """
    for idx in token_idxs:
        assert 0 <= idx < len(sent.tokens)

    phrase = Phrase()
    phrase.phrase_id = phrase_id
    phrase.sent = sent
    for token_idx in token_idxs:
        phrase.tokens.append(sent.tokens[token_idx].token_id)
    sent.phrases.append(phrase)


def load_phrases(input_file, lowercase=True):
    """
    The input phrase file is assumed to provide phrases in the following format
    per line:

    <phrase>\t<pos>

    <phrase> is a concatenation of words with '_'.
    """
    if input_file.endswith('.gz'):
        fin = gzip.open(input_file, 'rt')
    else:
        fin = open(input_file, 'r')

    phrase_dict = {}
    num_phrases = 0
    for line in fin:
        p = phrase_dict  # pointer
        items = line.rstrip('\n').split('\t')
        phrase = items[0]
        pos = items[1]
        for word in phrase.split('_'):
            if lowercase:
                word = word.lower()
            if word not in p:
                p[word] = {}
            p = p[word]
        p[PHRASE_END] = pos
        num_phrases += 1

    fin.close()
    print("Loaded {} phrases from {}".format(num_phrases, input_file))
    return phrase_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, metavar='str', default='out/sw100_wsd.json.gz',
                        help="input file (gzipped JSON)")
    parser.add_argument('--output', type=str, metavar='str', default='out/sw100_wsd_phrase.json.gz',
                        help="output file (gzipped JSON)")
    parser.add_argument('--deps', type=str, metavar='str', nargs='+',
                        default=['compound', 'advmod', 'xcomp', 'dobj'],
                        help="dependencies in discontinuous phrasal verbs")
    parser.add_argument('--phrases', type=str, metavar='str', default='resources/wordnet_phrases_pos.txt.gz',
                        help="input file for phrases")
    args = parser.parse_args()

    dataset = read_dataset(args.input)

    phrase_dict = load_phrases(args.phrases)

    print("Detecting phrases...")
    detect_phrases(dataset, phrase_dict, args.deps)

    print(f"Phrase detection is done.")

    write_dataset(dataset, args.output)
    print(f"Saved output to {args.output}")
