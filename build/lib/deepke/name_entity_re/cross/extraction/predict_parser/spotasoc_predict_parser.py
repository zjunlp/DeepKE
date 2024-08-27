#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import logging
from nltk.tree import ParentedTree
import re
from typing import Tuple, List, Dict


from ...extraction.constants import (
    null_span,
    type_start,
    type_end,
    span_start,
)
from ...extraction.predict_parser.predict_parser import PredictParser
from ...extraction.predict_parser.utils import fix_unk_from_text

logger = logging.getLogger(__name__)


left_bracket = '【'
right_bracket = '】'
brackets = left_bracket + right_bracket

split_bracket = re.compile(r"<extra_id_\d>")


def add_space(text):
    """
    add space between special token
    """
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]):
        new_text_list += item
    return ' '.join(new_text_list)


def convert_bracket(text):
    text = add_space(text)
    for start in [type_start]:
        text = text.replace(start, left_bracket)
    for end in [type_end]:
        text = text.replace(end, right_bracket)
    return text


def find_bracket_num(tree_str):
    """
    Count Bracket Number (num_left - num_right), 0 indicates num_left = num_right
    """
    count = 0
    for char in tree_str:
        if char == left_bracket:
            count += 1
        elif char == right_bracket:
            count -= 1
        else:
            pass
    return count


def check_well_form(tree_str):
    return find_bracket_num(tree_str) == 0


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()

    for index, char in enumerate(tree_str_list):
        if char == left_bracket:
            count += 1
            sum_count += 1
        elif char == right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def resplit_label_span(label, span, split_symbol=span_start):
    label_span = label + ' ' + span

    if split_symbol in label_span:
        splited_label_span = label_span.split(split_symbol)
        if len(splited_label_span) == 2:
            return splited_label_span[0].strip(), splited_label_span[1].strip()

    return label, span


def add_bracket(tree_str):
    """add right bracket to fix ill-formed expression
    """
    tree_str_list = tree_str.split()
    bracket_num = find_bracket_num(tree_str_list)
    tree_str_list += [right_bracket] * bracket_num
    return ' '.join(tree_str_list)


def get_tree_str(tree):
    """get str from sel tree
    """
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


def rewrite_label_span(label, span, label_set=None, text=None):

    # Invalid Type
    if label_set and label not in label_set:
        logger.debug('Invalid Label: %s' % label)
        return None, None

    # Fix unk using Text
    if text is not None and '<unk>' in span:
        span = fix_unk_from_text(span, text, '<unk>')

    # Invalid Text Span
    if text is not None and span not in text:
        logger.debug('Invalid Text Span: %s\n%s\n' % (span, text))
        return None, None

    return label, span


class SpotAsocPredictParser(PredictParser):

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None
               ) -> Tuple[List[Dict], Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_spot -> [(type1, text1), (type2, text2), ...]
                gold_spot -> [(type1, text1), (type2, text2), ...]
                pred_asoc -> [(spot type1, asoc type1, text1), (spot type2, asoc type2, text2), ...]
                gold_asoc -> [(spot type1, asoc type1, text1), (spot type2, asoc type2, text2), ...]
                pred_record -> [{'type': type1, 'text': text1, 'roles': [(spot type1, asoc type1, text1), ...]},
                                {'type': type2, 'text': text2, 'roles': [(spot type2, asoc type2, text2), ...]},
                                ]
                gold_record -> [{'type': type1, 'text': text1, 'roles': [(spot type1, asoc type1, text1), ...]},
                                {'type': type2, 'text': text2, 'roles': [(spot type2, asoc type2, text2), ...]},
                                ]
            Counter:
        """
        counter = Counter()
        well_formed_list = []

        if gold_list is None or len(gold_list) == 0:
            gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list)

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list,
                                              raw_list):
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)

            pred = clean_text(pred)

            try:
                gold_tree = ParentedTree.fromstring(gold, brackets=brackets)
            except ValueError:
                logger.warning(f"Ill gold: {gold}")
                logger.warning(f"Fix gold: {add_bracket(gold)}")
                gold_tree = ParentedTree.fromstring(
                    add_bracket(gold), brackets=brackets)
                counter.update(['gold_tree add_bracket'])

            instance = {
                'gold': gold,
                'pred': pred,
                'gold_tree': gold_tree,
                'text': text,
                'raw_data': raw_data
            }

            counter.update(['gold_tree' for _ in gold_tree])

            instance['gold_spot'], instance['gold_asoc'], instance['gold_record'] = self.get_record_list(
                sel_tree=instance["gold_tree"],
                text=instance['text']
            )

            try:
                if not check_well_form(pred):
                    pred = add_bracket(pred)
                    counter.update(['fixed'])

                pred_tree = ParentedTree.fromstring(pred, brackets=brackets)
                counter.update(['pred_tree' for _ in pred_tree])

                instance['pred_tree'] = pred_tree
                counter.update(['well-formed'])

            except ValueError:
                counter.update(['ill-formed'])
                logger.debug('ill-formed', pred)
                instance['pred_tree'] = ParentedTree.fromstring(
                    left_bracket + right_bracket,
                    brackets=brackets
                )

            instance['pred_spot'], instance['pred_asoc'], instance['pred_record'] = self.get_record_list(
                sel_tree=instance["pred_tree"],
                text=instance['text']
            )

            well_formed_list += [instance]

        return well_formed_list, counter

    def get_record_list(self, sel_tree, text=None):
        """ Convert single sel expression to extraction records
        Args:
            sel_tree (Tree): sel tree
            text (str, optional): _description_. Defaults to None.
        Returns:
            spot_list: list of (spot_type: str, spot_span: str)
            asoc_list: list of (spot_type: str, asoc_label: str, asoc_text: str)
            record_list: list of {'asocs': list(), 'type': spot_type, 'spot': spot_text}
        """

        spot_list = list()
        asoc_list = list()
        record_list = list()

        for spot_tree in sel_tree:

            # Drop incomplete tree
            if isinstance(spot_tree, str) or len(spot_tree) == 0:
                continue

            spot_type = spot_tree.label()
            spot_text = get_tree_str(spot_tree)
            spot_type, spot_text = resplit_label_span(
                spot_type, spot_text)
            spot_type, spot_text = rewrite_label_span(
                label=spot_type,
                span=spot_text,
                label_set=self.spot_set,
                text=text
            )

            # Drop empty generated span
            if spot_text is None or spot_text == null_span:
                continue
            # Drop empty generated type
            if spot_type is None:
                continue
            # Drop invalid spot type
            if self.spot_set is not None and spot_type not in self.spot_set:
                continue

            record = {'asocs': list(),
                      'type': spot_type,
                      'spot': spot_text}

            for asoc_tree in spot_tree:
                if isinstance(asoc_tree, str) or len(asoc_tree) < 1:
                    continue

                asoc_label = asoc_tree.label()
                asoc_text = get_tree_str(asoc_tree)
                asoc_label, asoc_text = resplit_label_span(
                    asoc_label, asoc_text)
                asoc_label, asoc_text = rewrite_label_span(
                    label=asoc_label,
                    span=asoc_text,
                    label_set=self.role_set,
                    text=text
                )

                # Drop empty generated span
                if asoc_text is None or asoc_text == null_span:
                    continue
                # Drop empty generated type
                if asoc_label is None:
                    continue
                # Drop invalid asoc type
                if self.role_set is not None and asoc_label not in self.role_set:
                    continue

                asoc_list += [(spot_type, asoc_label, asoc_text)]
                record['asocs'] += [(asoc_label, asoc_text)]

            spot_list += [(spot_type, spot_text)]
            record_list += [record]

        return spot_list, asoc_list, record_list
