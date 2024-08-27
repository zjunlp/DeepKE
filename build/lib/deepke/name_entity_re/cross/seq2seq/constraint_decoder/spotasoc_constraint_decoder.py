#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from typing import List, Dict
from ...extraction.label_tree import get_label_name_tree
from ...extraction.constants import (
    span_start,
    type_start,
    type_end,
    null_span,
    text_start
)
from ...seq2seq.constraint_decoder.constraint_decoder import (
    ConstraintDecoder,
    find_bracket_position,
    generated_search_src_sequence
)


debug = True if 'DEBUG' in os.environ else False


class SpotAsocConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = self.tokenizer.convert_tokens_to_ids([span_start])[0]
        self.type_tree = get_label_name_tree(type_schema.type_list, self.tokenizer, end_symbol=self.tree_end)
        self.role_tree = get_label_name_tree(type_schema.role_list, self.tokenizer, end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([type_start])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([type_end])[0]
        self.span_start = self.tokenizer.convert_tokens_to_ids([span_start])[0]
        self.null_span = self.tokenizer.convert_tokens_to_ids([null_span])[0]
        self.text_start = self.tokenizer.convert_tokens_to_ids([text_start])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        # special_token_set = {EVENT_TYPE_LEFT, EVENT_TYPE_RIGHT}
        special_token_set = {self.type_start, self.type_end, self.span_start}
        special_index_token = list(filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(bracket_position[self.type_end])

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_trigger'
            if last_special_token == self.span_start:
                state = 'generate_trigger_text'
        elif start_number == end_number + 3:
            state = 'generate_role'
            if last_special_token == self.span_start:
                state = 'generate_role_text'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(self.tokenizer.eos_token_id)]

        if self.text_start in src_sentence:
            src_sentence = src_sentence[src_sentence.index(self.text_start) + 1:]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Decode Error:")
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_trigger':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                # EVENT_TYPE_LEFT: Start a new role
                # EVENT_TYPE_RIGHT: End this event
                return [self.type_start, self.type_end]
            else:
                valid_tokens = self.search_prefix_tree(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    end_search_tokens=[self.span_start]
                )

        elif state in {'generate_trigger_text'}:
            generated = tgt_generated[index + 1:]

            if len(generated) > 0 and generated[-1] == self.null_span:
                return [self.type_end, self.type_start]

            valid_tokens = generated_search_src_sequence(
                generated=generated,
                src_sequence=src_sentence + [self.null_span],
                end_sequence_search_tokens=[self.type_end, self.type_start],
            )

        elif state in {'generate_role_text'}:
            generated = tgt_generated[index + 1:]

            if len(generated) > 0 and generated[-1] == self.null_span:
                return [self.type_end]

            valid_tokens = generated_search_src_sequence(
                generated=generated,
                src_sequence=src_sentence + [self.null_span],
                end_sequence_search_tokens=[self.type_end],
            )

        elif state == 'generate_role':

            if tgt_generated[-1] == self.type_start:
                # Start Role Label
                return list(self.role_tree.keys())

            generated = tgt_generated[index + 1:]
            valid_tokens = self.search_prefix_tree(
                generated=generated,
                prefix_tree=self.role_tree,
                end_search_tokens=[self.span_start]
            )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError('State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % self.tokenizer.convert_ids_to_tokens(valid_tokens)) if debug else None
        return valid_tokens

    def search_prefix_tree(self, generated: List[str], prefix_tree: Dict,
                           end_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                return end_search_tokens

        valid_token = list(tree.keys())
        if self.tree_end in valid_token:
            valid_token.remove(self.tree_end)
            valid_token += end_search_tokens
        return valid_token


class SpotConstraintDecoder(SpotAsocConstraintDecoder):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        special_token_set = {self.type_start, self.type_end, self.span_start}
        special_index_token = list(filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(bracket_position[self.type_end])

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_span'
            if last_special_token == self.span_start:
                state = 'generate_span_text'
        else:
            state = 'error'
        return state, last_special_index

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(self.tokenizer.eos_token_id)]

        if self.text_start in src_sentence:
            src_sentence = src_sentence[src_sentence.index(self.text_start) + 1:]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Decode Error:")
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_span':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                raise RuntimeError('Invalid %s in %s' % (self.type_end, tgt_generated))

            else:
                valid_tokens = self.search_prefix_tree(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    end_search_tokens=[self.span_start]
                )

        elif state == 'generate_span_text':
            generated = tgt_generated[index + 1:]
            valid_tokens = generated_search_src_sequence(
                generated=generated,
                src_sequence=src_sentence + [self.null_span],
                end_sequence_search_tokens=[self.type_end],
            )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError('State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % valid_tokens) if debug else None
        return valid_tokens
