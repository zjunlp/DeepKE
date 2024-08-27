#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict
import os
from typing import List


def match_sublist(the_list, to_match):
    """

    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match:
        [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def find_bracket_position(generated_text, _type_start, _type_end):
    bracket_position = {_type_start: list(), _type_end: list()}
    for index, char in enumerate(generated_text):
        if char in bracket_position:
            bracket_position[char] += [index]
    return bracket_position


def build_sentence_tree(sentence):
    tree = defaultdict(set)

    for prev_token, next_token in zip(sentence[:-1], sentence[1:]):
        tree[prev_token].add(next_token)

    return tree


def generated_search_prefix_tree(generated, prefix_tree, tokenizer):
    tree = prefix_tree
    # Leaf is KEY_VALUE_SPLIT
    for token in generated:

        if token not in tree:
            return [tokenizer.eos_token]
        tree = tree[token]

    return list(tree)


def generated_search_src_sequence(generated, src_sequence, end_sequence_search_tokens=None):

    if len(generated) == 0:
        # All src tokens are valid before generation
        return src_sequence

    matched_tuples = match_sublist(the_list=src_sequence, to_match=generated)

    valid_token = list()
    for _, end in matched_tuples:
        next_index = end + 1
        if next_index < len(src_sequence):
            valid_token += [src_sequence[next_index]]

    if end_sequence_search_tokens:
        valid_token += end_sequence_search_tokens

    return valid_token


class ConstraintDecoder:
    def __init__(self, tokenizer, source_prefix):
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.source_prefix_tokenized = tokenizer.encode(source_prefix,
                                                        add_special_tokens=False) if source_prefix else []

    def get_state_valid_tokens(self, src_sentence: List[str], tgt_generated: List[str]) -> List[str]:
        pass

    def constraint_decoding(self, src_sentence, tgt_generated):
        if self.source_prefix_tokenized:
            # Remove Source Prefix for Generation
            src_sentence = src_sentence[len(self.source_prefix_tokenized):]

        valid_token_ids = self.get_state_valid_tokens(src_sentence.tolist(), tgt_generated.tolist())

        return valid_token_ids
