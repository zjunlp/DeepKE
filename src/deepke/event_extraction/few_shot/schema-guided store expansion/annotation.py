#!/usr/bin/env python

import json
import sys
import gzip
import pdb


def write_dataset(dataset, output_file):
    with gzip.open(output_file, 'wt') as fout:
        json.dump(dataset.to_dict(), fout, indent=2)


def read_dataset(input_file):
    with gzip.open(input_file, 'rt') as fin:
        dataset = Dataset()
        dataset.from_dict(json.load(fin))
    return dataset


def get_head_word_index(tokens):
    """
    Input:
    tokens (list) :: list of tokens
    """
    if not tokens:
        return None
    num_tokens = len(tokens)
    if num_tokens == 1:
        return 0
    ordered_indexes = []  # indexes of tokens sorted with the head-dependency order
    for i in range(num_tokens):
        token1 = tokens[i]
        for j in range(i+1, num_tokens):
            token2 = tokens[j]
            if token1.is_dependency_ancestor(token2):
                if i in ordered_indexes:
                    index1 = ordered_indexes.index(i)
                    if j in ordered_indexes:
                        index2 = ordered_indexes.index(j)
                        if index1 > index2:  # need to reverse the order
                            ordered_indexes[index1] = j
                            ordered_indexes[index2] = i
                    else:
                        ordered_indexes.insert(index1 + 1, j)
                else:
                    if j in ordered_indexes:
                        ordered_indexes.insert(max(ordered_indexes.index(j) - 1, 0), i)
                    else:
                        ordered_indexes.append(i)
                        ordered_indexes.append(j)
            elif token2.is_dependency_ancestor(token1):
                if i in ordered_indexes:
                    index1 = ordered_indexes.index(i)
                    if j in ordered_indexes:
                        index2 = ordered_indexes.index(j)
                        if index1 < index2:  # need to reverse the order
                            ordered_indexes[index1] = j
                            ordered_indexes[index2] = i
                    else:
                        ordered_indexes.insert(max(index1 - 1, 0), j)
                else:
                    if j in ordered_indexes:
                        ordered_indexes.insert(ordered_indexes.index(j) + 1, i)
                    else:
                        ordered_indexes.append(j)
                        ordered_indexes.append(i)
    if len(ordered_indexes) > 1:
        return ordered_indexes[0]
    return None


class JSONSerializable(object):
    def __init__(self):
        self.name = self.__class__.__name__

    def from_dict(self, d, parent=None):
        current_module = sys.modules[__name__]
        key2vtype = {k:type(v).__name__ for k, v in self.__dict__.items()}
        for k, v in key2vtype.items():
            if k == 'name':
                continue
            if k == 'doc' or k == 'sent':
                setattr(self, k, parent)
            if k not in d:
                continue
            if v == 'list':
                values = getattr(self, k)
                for item in d[k]:
                    if isinstance(item, dict):
                        _class = getattr(current_module, item['name'])
                        class_ins = _class()
                        class_ins.from_dict(item, self)
                        values.append(class_ins)
                    else:
                        values.append(item)
            else:
                setattr(self, k, d[k])

    def to_dict(self):
        return {k:[item.to_dict() if isinstance(item, JSONSerializable) else item for item in v]
                if isinstance(v, list) else v
                for k, v in self.__dict__.items() if isinstance(v, (list, str, int, float))}


class Dataset(JSONSerializable):
    def __init__(self):
        super().__init__()
        self.docs = []


class Document(JSONSerializable):
    def __init__(self):
        super().__init__()
        self.doc_id = None
        self.text = None
        self.sents = []

    def __repr__(self):
        return f'Document(id={self.doc_id},text={self.text})'


class Sentence(JSONSerializable):
    def __init__(self):
        super().__init__()
        self.sent_id = None
        self.begin = None
        self.end = None
        self.tokens = []
        self.phrases = []
        self.event_nuggets = []
        self.doc = None

    @property
    def text(self):
        return self.doc.text[self.begin:self.end]

    def __repr__(self):
        return f'Sentence(id={self.sent_id},begin={self.begin},end={self.end},text={self.text})'


class Token(JSONSerializable):
    def __init__(self):
        super().__init__()
        self.token_id = None
        self.pos = None
        self.lemma = None
        self.begin = None
        self.end = None
        self.is_root_dep_node = None  # whether this token is a root dependency node
        self.sense_id = None
        # Dependency relations to heads of this token (could be multiple).
        self.head_deps = []
        # Dependency relations to children of this token.
        self.tail_deps = []
        self.sent = None

    @property
    def text(self):
        return self.sent.doc.text[self.begin:self.end]

    def __repr__(self):
        return f'Token(id={self.token_id},begin={self.begin},end={self.end},text={self.text})'

    def is_dependency_ancestor(self, other):
        """
        Returns true if this token is a dependency ancestor of the given token.
        """
        # Breadth-first search implementation
        visited, queue = set(), [other]
        while queue:
            curr_token = queue.pop(0)
            if curr_token is self:
                return True
            for head_dep in curr_token.head_deps:
                if head_dep.head is None:  # dependency root
                    continue
                head = curr_token.sent.tokens[head_dep.head-1]
                if head not in visited:
                    visited.add(head)
                    queue.append(head)
        return False


class Phrase(JSONSerializable):
    def __init__(self):
        super().__init__()
        self.phrase_id = None
        self.tokens = []  # a list of token IDs (not token objects)
        self.sent = None

    @property
    def text(self):
        return ' '.join([self.sent.tokens[token_id-1].text for token_id in self.tokens])

    def __repr__(self):
        return f'Phrase(id={self.phrase_id},text={self.text})'


class Dependency(JSONSerializable):
    def __init__(self):
        super().__init__()
        self.rel = None  # dependency relation type
        self.head = None
        self.tail = None

    def __repr__(self):
        return f'Dependency(rel={self.rel},head={self.head},tail={self.tail})'


class EventNugget(JSONSerializable):
    def __init__(self):
        super().__init__()
        self.nugget_id = None
        self.tokens = []
        self.sent = None

    @property
    def text(self):
        return ' '.join([self.sent.tokens[token_id-1].text for token_id in self.tokens])

    def __repr__(self):
        return f'EventNugget(id={self.nugget_id},text={self.text})'
