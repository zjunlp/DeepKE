"""
This script extracts IE annotations from ACE2005 (LDC2006T06).

Usage:
python process_ace.py \
    
"""
import os
import re
import json
import glob
import tqdm
import random
import torch
from lxml import etree
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from dataclasses import dataclass
from argparse import ArgumentParser
from transformers import (BertTokenizer,
                          RobertaTokenizer,
                          XLMRobertaTokenizer,
                          PreTrainedTokenizer,
                          AutoTokenizer)
from nltk import (sent_tokenize as sent_tokenize_,
                  wordpunct_tokenize as wordpunct_tokenize_)
import jieba                  
import stanza

nlp_ar = stanza.Pipeline(lang='ar', processors='tokenize')
nlp_en = stanza.Pipeline(lang='en', processors='tokenize')
nlp_zh = stanza.Pipeline(lang='zh', processors='tokenize')

TAG_PATTERN = re.compile('<[^<>]+>', re.MULTILINE)

DOCS_TO_REVISE_SENT = {
    'CNN_ENG_20030529_130011.6': [(461, 504),
                                  (668, 859),
                                  (984, 1074),
                                  (1577, 1632)],
    'CNN_ENG_20030626_203133.11': [(1497, 1527)],
    'CNN_ENG_20030526_180540.6': [(67, 99)],
    'CNNHL_ENG_20030523_221118.14': [(136, 174)],
    'BACONSREBELLION_20050127.1017': [(2659, 2663),
                                      (4381, 4405),
                                      (410, 458)],
    'misc.legal.moderated_20050129.2225': [(4118, 4127),
                                           (4710, 4794)],
    'alt.vacation.las-vegas_20050109.0133': [(1201, 1248)],
    'alt.obituaries_20041121.1339': [(1947, 2044), (1731, 1737)],
    'APW_ENG_20030326.0190': [(638, 739)],
    'APW_ENG_20030403.0862': [(729, 781)],
    'CNN_IP_20030405.1600.02': [(699, 705)],
    'CNN_IP_20030403.1600.00-1': [(2392, 2399)],
    'CNN_IP_20030409.1600.04': [(1039, 1050)],
    'CNN_IP_20030412.1600.03': [(741, 772)],
    'CNN_IP_20030402.1600.02-1': [(885, 892)],
    'CNN_IP_20030329.1600.02': [(3229, 3235)],
    'CNN_IP_20030409.1600.02': [(477, 498)],
    'CNN_CF_20030304.1900.04': [(522, 575),
                                (5193, 5210),
                                (5461, 5542)],
    'CNN_IP_20030403.1600.00-3': [(1487, 1493)],
    'soc.history.war.world-war-ii_20050127.2403': [(414, 441)],
    'CNN_ENG_20030529_130011.6': [(209, 254),
                                  (461, 504),
                                  (668, 859),
                                  (984, 1074),
                                  (1577, 1632)],
    }

SKIPPED_DOCS = [
    'ALFILFILM_20050202.0740'
]

def mask_escape(text: str) -> str:
    """Replaces escaped characters with rare sequences.

    Args:
        text (str): text to mask.
    
    Returns:
        str: masked string.
    """
    return text.replace('&amp;', 'ҪҪҪҪҪ').replace('&lt;', 'ҚҚҚҚ').replace('&gt;', 'ҺҺҺҺ')


def unmask_escape(text: str) -> str:
    """Replaces masking sequences with the original escaped characters.

    Args:
        text (str): masked string.
    
    Returns:
        str: unmasked string.
    """
    return text.replace('ҪҪҪҪҪ', '&amp;').replace('ҚҚҚҚ', '&lt;').replace('ҺҺҺҺ', '&gt;')


def recover_escape(text: str) -> str:
    """Converts named character references in the given string to the corresponding
    Unicode characters. I didn't notice any numeric character references in this
    dataset.

    Args:
        text (str): text to unescape.
    
    Returns:
        str: unescaped string.
    """
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')


def sent_tokenize(text: Tuple[str, int, int],
                  language: str = 'english') -> List[Tuple[str, int, int]]:
    """
    Performs sentence tokenization using stanza

    Args:
        text (Tuple[str, int, int]): a tuple of three elements, text to split 
            into sentences, start offset, and end offset. 
        language (str): available options: english, chinese, arabic.
    
    Returns:
        List[Tuple[str, int, int]]: a list of sentences.
    """
    text, start, end = text
    if language == 'english':
        sentences = sent_tokenize_(text, language=language)
    else:
        if language == 'chinese':
            doc = nlp_zh(text)
        elif language == 'arabic':
            doc = nlp_ar(text)
        
        ending_char_idx = [0]
        for sent in doc.sentences:
            ending_char_idx.append(sent.tokens[-1].end_char)
        ending_char_idx = ending_char_idx[:-1]
        ending_char_idx.append(len(text))
        sentences = []
        for idx in range(1, len(ending_char_idx)):
            sentences.append(text[ending_char_idx[idx-1]: ending_char_idx[idx]])

    last = 0
    sentences_ = []
    for sent in sentences:
        index = text[last:].find(sent)
        if index == -1:
            print(text, sent)
        else:
            sentences_.append((sent, last + index + start,
                               last + index + len(sent) + start))
        last += index + len(sent)
    return sentences_


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace('\n', ' ')

    def char_offsets_to_token_offsets(self, tokens: List[Tuple[int, int, str]]):
        """Converts self.start and self.end from character offsets to token
        offsets.

        Args:
            tokens (List[int, int, str]): a list of token tuples. Each item in
                the list is a triple (start_offset, end_offset, text).
        """
        start_ = end_ = -1
        for i, (s, e, _) in enumerate(tokens):
            if s == self.start:
                start_ = i
            if e == self.end:
                end_ = i + 1
        if start_ == -1 or end_ == -1 or start_ > end_:
            raise ValueError('Failed to update offsets for {}-{}:{} in {}'.format(
                self.start, self.end, self.text, tokens))
        self.start, self.end = start_, end_

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            dict: a dict of instance variables.
        """
        return {
            'text': recover_escape(self.text),
            'start': self.start,
            'end': self.end,
        }

    def remove_space(self):
        """Removes heading and trailing spaces in the span text."""
        # heading spaces
        text = self.text.lstrip(' ')
        self.start += len(self.text) - len(text)
        # trailing spaces
        after_text = text.rstrip(' ')
        #self.end = self.start + len(text)
        #self.text = text
        self.end += len(after_text) - len(text)
        self.text = after_text


    def copy(self):
        """Makes a copy of itself.

        Returns:
            Span: a copy of itself."""
        return Span(self.start, self.end, self.text)


@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    entity_subtype: str
    mention_type: str
    value: str = None

    def copy(self):
        return Entity(self.start, self.end, self.text,
            self.entity_id, self.mention_id, self.entity_type,
            self.entity_subtype, self.mention_type, self.value)

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            'text': recover_escape(self.text),
            'entity_id': self.entity_id,
            'mention_id': self.mention_id,
            'start': self.start,
            'end': self.end,
            'entity_type': self.entity_type,
            'entity_subtype': self.entity_subtype,
            'mention_type': self.mention_type,
        }
        if self.value:
            entity_dict['value'] = self.value
        return entity_dict

@dataclass
class Entity_whole:
    entity_id: str
    entity_type: str
    entity_subtype: str
    entity_mentions: List[Entity]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'entity_subtype': self.entity_subtype,
            'entity_mentions': [e.to_dict() for e in self.entity_mentions]
        }
        return entity_dict

    def copy(self):
        """Makes a copy of itself.
        """
        return Entity_whole(self.entity_id, self.entity_type, 
                            self.entity_subtype, self.entity_mentions)


@dataclass
class RelationArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'mention_id': self.mention_id,
            'role': self.role,
            'text': recover_escape(self.text)
        }

@dataclass
class RelEntBaseArgument:
    entity_id: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'entity_id': self.entity_id,
            'role': self.role,
        }

@dataclass
class Relation:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type,
            'relation_subtype': self.relation_subtype,
            'arg1': self.arg1.to_dict(),
            'arg2': self.arg2.to_dict(),
        }

@dataclass
class RelEntBase:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelEntBaseArgument
    arg2: RelEntBaseArgument

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type,
            'relation_subtype': self.relation_subtype,
            'arg1': self.arg1.to_dict(),
            'arg2': self.arg2.to_dict(),
        }

@dataclass
class EventArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'mention_id': self.mention_id,
            'role': self.role,
            'text': recover_escape(self.text),
        }

@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: List[EventArgument]

    def copy(self):
        return Event(self.event_id, self.mention_id, self.event_type,
                self.event_subtype, self.trigger.copy(), self.arguments)

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'event_id': self.event_id,
            'mention_id': self.mention_id,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            'trigger': self.trigger.to_dict(),
            'arguments': [arg.to_dict() for arg in self.arguments],
        }

@dataclass
class EventArgument_whole:
    entity_id: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'entity_id': self.entity_id,
            'role': self.role,
        }

@dataclass
class Event_whole:
    event_id: str
    event_type: str
    event_subtype: str
    arguments: List[EventArgument_whole]
    event_mentions: List[Event]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            'arguments': [arg.to_dict() for arg in self.arguments],
            'event_mentions': [e.to_dict() for e in self.event_mentions]
        }       


@dataclass
class Sentence(Span):
    sent_id: str
    tokens: List[str]
    sent_starts: List[int]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]
    entity_cluster: List[Entity_whole]
    relations_entbase: List[RelEntBase]
    event_cluster: List[Event_whole]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'sent_id': self.sent_id,
            'tokens': [recover_escape(t) for t in self.tokens],
            'sentence_starts': self.sent_starts,
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations],
            'events': [event.to_dict() for event in self.events],
            'entity_cluster': [entity.to_dict() for entity in self.entity_cluster],
            'event_cluster': [event.to_dict() for event in self.event_cluster],
            'relations_entbase': [relation.to_dict() for relation in self.relations_entbase],
            'start': self.start,
            'end': self.end,
            'text': recover_escape(self.text).replace('\t', ' '),
        }


@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'doc_id': self.doc_id,
            'sentences': [sent.to_dict() for sent in self.sentences]
        }


def revise_sentences(sentences: List[Tuple[str, int, int]],
                     doc_id: str) -> List[Tuple[int, int, str]]:
    """Automatic sentence tokenization may have errors for a few documents.

    Args:
        sentences (List[Tuple[str, int, int]]): a list of sentence tuples.
        doc_id (str): document ID.

    Returns:
        List[Tuple[str, int, int]]: a list of revised sentence tuples.
    """
    sentences_ = []

    offset_list = DOCS_TO_REVISE_SENT[doc_id]
    first_part_offsets = {offset for offset, _ in offset_list}
    second_part_offsets = {offset for _, offset in offset_list}


    for sentence_idx, (text, start, end) in enumerate(sentences):
        if start in first_part_offsets:
            next_text, next_start, next_end = sentences[sentence_idx + 1]
            space = ' ' * (next_start - end)
            sentences_.append((text + space + next_text, start, next_end))
        elif start in second_part_offsets:
            continue
        else:
            sentences_.append((text, start, end))
    
    return sentences_


def read_sgm_file(path: str,
                  language: str = 'english') -> List[Tuple[str, int, int]]:
    """Reads a SGM text file.
    
    Args:
        path (str): path to the input file.
        language (str): document language. Valid values: "english", "arabic" or "chinese".

    Returns:
        List[Tuple[str, int, int]]: a list of sentences. Each item in the list
            is a tuple of three elements, sentence text, start offset, and end
            offset.
    """
    data = open(path, 'r', encoding='utf-8').read()

    # Chunk the document
    chunks = TAG_PATTERN.sub('⁑', data).split('⁑')

    # Get the offset of <TEXT>
    data = data.replace('<TEXT>', '⁂')
    data = TAG_PATTERN.sub('', data)
    min_offset = max(0, data.find('⁂'))
    data = data.replace('⁂', '')

    # Extract sentences from chunks
    chunk_offset = 0
    sentences = []
    for chunk in chunks:
        lines = chunk.split('\n')
        current_sentence = []
        start = offset = 0
        for line in lines:
            offset += len(line) + 1
            if line.strip():
                current_sentence.append(line)
            else:
                # empty line
                if current_sentence:
                    sentence = ' '.join(current_sentence)
                    if start + chunk_offset >= min_offset:
                        sentences.append((sentence,
                                          start + chunk_offset,
                                          start + chunk_offset + len(sentence)))
                    current_sentence = []
                start = offset
        if current_sentence:
            sentence = ' '.join(current_sentence)
            if start + chunk_offset >= min_offset:
                sentences.append((sentence,
                                  start + chunk_offset,
                                  start + chunk_offset + len(sentence)))
        chunk_offset += len(chunk)

    # Re-tokenize sentences
    sentences = [s for sent in sentences
                 for s in sent_tokenize(sent, language=language)]

    return sentences

def read_apf_file(path: str,
                  time_and_val: bool = False
                 ) -> Tuple[str, str, List[Entity], List[Relation], List[Event]]:
    """Reads an APF file.

    Args:
        path (str): path to the input file.
        time_and_val (bool): extract times and values or not.
    
    Returns:
        doc_id (str): document ID.
        source (str): document source.
        entity_list (List[Entity]): a list of Entity instances.
        relation_list (List[Relation]): a list of Relation instances.
        event_list (List[Event]): a list of Events instances.
    """
    data = open(path, 'r', encoding='utf-8').read()
    soup = BeautifulSoup(data, 'lxml-xml')

    # metadata
    root = soup.find('source_file')
    source = root['SOURCE']
    doc = root.find('document')
    doc_id = doc['DOCID']

    entity_list, relation_list, event_list = [], [], []
    entity_clusters, event_clusters, relentbase_list = [], [], []
    

    # entities: nam, nom, pro
    for entity in doc.find_all('entity'):
        entity_id = entity['ID']
        entity_type = entity['TYPE']
        entity_subtype = entity['SUBTYPE']
        entity_cluster = []
        for entity_mention in entity.find_all('entity_mention'):
            mention_id = entity_mention['ID']
            mention_type = entity_mention['TYPE']
            head = entity_mention.find('head').find('charseq')
            start, end, text = int(head['START']), int(head['END']), head.text
            entity_list.append(Entity(start, end+1, text,
                                      entity_id, mention_id, entity_type,
                                      entity_subtype, mention_type))
            entity_cluster.append(Entity(start, end+1, text,
                                      entity_id, mention_id, entity_type,
                                      entity_subtype, mention_type))
        entity_clusters.append(Entity_whole(entity_id, entity_type, entity_subtype, entity_cluster))
    if time_and_val:
        # entities: value
        for entity in doc.find_all('value'):
            enitty_id = entity['ID']
            entity_type = entity['TYPE']
            entity_subtype = entity.get('SUBTYPE', None)
            entity_cluster = []
            for entity_mention in entity.find_all('value_mention'):
                mention_id = entity_mention['ID']
                mention_type = 'VALUE'
                extent = entity_mention.find('extent').find('charseq')
                start, end, text = int(extent['START']), int(
                    extent['END']), extent.text
                entity_list.append(Entity(start, end+1, text,
                                          entity_id, mention_id, entity_type,
                                          entity_subtype, mention_type))
                entity_cluster.append(Entity(start, end+1, text,
                                            entity_id, mention_id, entity_type,
                                            entity_subtype, mention_type))
            entity_clusters.append(Entity_whole(entity_id, entity_type, entity_subtype, entity_cluster))

        # entities: timex
        for entity in doc.find_all('timex2'):
            entity_id = entity['ID']
            enitty_type = entity_subtype = 'TIME'
            value = entity.get('VAL', None)
            entity_cluster = []
            for entity_mention in entity.find_all('timex2_mention'):
                mention_id = entity_mention['ID']
                mention_type = 'TIME'
                extent = entity_mention.find('extent').find('charseq')
                start, end, text = int(extent['START']), int(
                    extent['END']), extent.text
                entity_list.append(Entity(start, end+1, text,
                                          entity_id, mention_id, entity_type,
                                          entity_subtype, mention_type,
                                          value=value))
                entity_cluster.append(Entity(start, end+1, text,
                                          entity_id, mention_id, entity_type,
                                          entity_subtype, mention_type,
                                          value=value))
            entity_clusters.append(Entity_whole(entity_id, entity_type, entity_subtype, entity_cluster))
    
    # relations
    for relation in doc.find_all('relation'):
        relation_id = relation['ID']
        relation_type = relation['TYPE']
        if relation_type == 'METONYMY':
            continue
        relation_subtype = relation['SUBTYPE']
        for relation_mention in relation.find_all('relation_mention'):
            mention_id = relation_mention['ID']
            arg1 = arg2 = None
            for arg in relation_mention.find_all('relation_mention_argument'):
                arg_mention_id = arg['REFID']
                arg_role = arg['ROLE']
                arg_text = arg.find('extent').find('charseq').text
                if arg_role == 'Arg-1':
                    arg1 = RelationArgument(arg_mention_id, arg_role, arg_text)
                elif arg_role == 'Arg-2':
                    arg2 = RelationArgument(arg_mention_id, arg_role, arg_text)
            if arg1 and arg2:
                relation_list.append(Relation(mention_id, relation_type,
                                              relation_subtype, arg1, arg2))
        arg1 = arg2 = None
        for relation_argument in relation.find_all('relation_argument'):
            arg_id = arg['REFID']
            arg_role = arg['ROLE']
            if arg_role == 'Arg-1':
                arg1 = RelEntBaseArgument(arg_id, arg_role)
            elif arg_role == 'Arg-2':
                arg2 = RelEntBaseArgument(arg_id, arg_role)
        if arg1 and arg2:
           relentbase_list.append(RelEntBase(relation_id, relation_type, relation_subtype, 
                                            arg1, arg2)) 

    # events
    for event in doc.find_all('event'):
        event_id = event['ID']
        event_type = event['TYPE']
        event_subtype = event['SUBTYPE']
        event_modality = event['MODALITY']
        event_polarity = event['POLARITY']
        event_genericity = event['GENERICITY']
        event_tense = event['TENSE']
        event_cluster = []
        for event_mention in event.find_all('event_mention'):
            mention_id = event_mention['ID']
            trigger = event_mention.find('anchor').find('charseq')
            trigger_start, trigger_end = int(
                trigger['START']), int(trigger['END'])
            trigger_text = trigger.text
            event_args = []
            for arg in event_mention.find_all('event_mention_argument'):
                arg_mention_id = arg['REFID']
                arg_role = arg['ROLE']
                arg_text = arg.find('extent').find('charseq').text
                event_args.append(EventArgument(
                    arg_mention_id, arg_role, arg_text))
            event_list.append(Event(event_id, mention_id,
                                    event_type, event_subtype,
                                    Span(trigger_start,
                                         trigger_end + 1, trigger_text),
                                    event_args))
            event_cluster.append(Event(event_id, mention_id,
                                    event_type, event_subtype,
                                    Span(trigger_start,
                                         trigger_end + 1, trigger_text),
                                    event_args))
        event_arguments = []
        for arg in event.find_all('event_argument'):
            arg_id = arg['REFID']
            arg_role = arg['ROLE']
            event_arguments.append(EventArgument_whole(arg_id, arg_role))
        event_clusters.append(Event_whole(event_id, event_type, event_subtype, 
                                        event_arguments, event_cluster))

    # remove heading/tailing spaces
    for entity in entity_list:
        entity.remove_space()
    for event in event_list:
        event.trigger.remove_space()
    for entities in entity_clusters:
        for entity in entities.entity_mentions:
            entity.remove_space()
    for events in event_clusters:
        for event in events.event_mentions:
            event.trigger.remove_space()

    return (doc_id, source, entity_list, relation_list, event_list, 
        entity_clusters, event_clusters, relentbase_list)


def process_entities(entities: List[Entity],
                     sentences: List[List[Tuple[str, int, int]]]
                    ) -> List[List[Entity]]:
    """Cleans entities and splits them into lists

    Args:
        entities (List[Entity]): a list of Entity instances.
        sentences (List[List[Tuple[str, int, int]]]): a list of window of sentences.

    Returns:
        List[List[Entity]]: a list of sentence entity lists.
    """
    sentence_entities = [[] for _ in range(len(sentences))]

    # assign each entity to the sentence where it appears
    for entity in entities:
        start, end = entity.start, entity.end
        for i, sentence in enumerate(sentences): # each window
            flag = False
            for text, s, e in sentence:
                if start >= s and end <= e:
                    # put this entity into this window
                    flag = True
                    break
            if flag:
                sentence_entities[i].append(entity.copy())
            # TODO: I'm not sure whether there will be some entities that 
            # cross single sentence boundary
    
    """
    # remove overlapping entities
    # TODO: if you want to use it, you need to modify some in order to fit "window" setting
    sentence_entities_cleaned = [[] for _ in range(len(sentences))]
    for i, entities in enumerate(sentence_entities):
        if not entities:
            continue
        # prefer longer entities
        entities.sort(key=lambda x: (x.end - x.start), reverse=True)
        chars = [0] * max([x.end for x in entities])
        for entity in entities:
            overlap = False
            for j in range(entity.start, entity.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if not overlap:
                chars[entity.start:entity.end] = [
                    1] * (entity.end - entity.start)
                sentence_entities_cleaned[i].append(entity)
        sentence_entities_cleaned[i].sort(key=lambda x: x.start)
    """
    return sentence_entities


def process_entity_cluster(entity_cluster: List[Entity_whole],
                           sentences: List[List[Tuple[str, int, int]]]
                           ) -> List[List[Entity_whole]]:
    """Cleans entity_cluster and splits them into lists

    Args:
        entities (List[Entity_whole]): a list of Entity_whole instances.
        sentences (List[List[Tuple[str, int, int]]]): a list of window of sentences.

    Returns:
        List[List[Entity_whole]]: a list of sentence Entity_whole lists.
    """
    sentence_entities = [[] for _ in range(len(sentences))]

    # assign each entity_whole to the sentence where it appears
    for i, sentence in enumerate(sentences): # each window of sentence
        for entity in entity_cluster:
            new_mention_list = []
            for mention in entity.entity_mentions:
                start, end = mention.start, mention.end
                for _, s, e in sentence:
                    if start >= s and end <= e:
                        # it means we can find this mention in this window
                        new_mention_list.append(mention.copy())
                        break
            if any(new_mention_list):
                new_entity_whole = entity.copy()
                new_entity_whole.entity_mentions = new_mention_list
                sentence_entities[i].append(new_entity_whole)
    return sentence_entities


def process_events(events: List[Event],
                   sentence_entities: List[List[Entity]],
                   sentences: List[List[Tuple[str, int, int]]]
                  ) -> List[List[Event]]:
    """Cleans and assigns events.

    Args:
        events (List[Event]): A list of Event objects
        entence_entities (List[List[Entity]]): A list of sentence entity lists.
        sentences (List[List[Tuple[str, int, int]]]): a list of window of sentences.
    
    Returns:
        List[List[Event]]: a list of sentence event lists.
    """
    sentence_events = [[] for _ in range(len(sentences))]
    # assign each event mention to the sentence where it appears
    for event in events:
        start, end = event.trigger.start, event.trigger.end
        for i, sentence in enumerate(sentences): # each window
            # check trigger
            flag = False
            for _, s, e in sentence: # every sentence in this window
                if start >= s and end <= e:
                    flag = True
                    break
            if flag:
                sent_entities = sentence_entities[i]
                # clean the argument list
                arguments = []
                for argument in event.arguments:
                    mention_id = argument.mention_id
                    for entity in sent_entities:
                        if entity.mention_id == mention_id:
                            arguments.append(argument)
                event_cleaned = Event(event.event_id, event.mention_id,
                                    event.event_type, event.event_subtype,
                                    trigger=event.trigger.copy(),
                                    arguments=arguments)
                sentence_events[i].append(event_cleaned)
    """
    # remove overlapping events
    # TODO: if you want to use it, you need to modify some in order to fit "window" setting
    sentence_events_cleaned = [[] for _ in range(len(sentences))]
    for i, events in enumerate(sentence_events):
        if not events:
            continue
        events.sort(key=lambda x: (x.trigger.end - x.trigger.start),
                    reverse=True)
        chars = [0] * max([x.trigger.end for x in events])
        for event in events:
            overlap = False
            for j in range(event.trigger.start, event.trigger.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if not overlap:
                chars[event.trigger.start:event.trigger.end] = [
                    1] * (event.trigger.end - event.trigger.start)
                sentence_events_cleaned[i].append(event)
        sentence_events_cleaned[i].sort(key=lambda x: x.trigger.start)
    """
    return sentence_events


def process_event_cluster(event_clusters: List[Event_whole],
                        sentence_events: List[List[Event]],
                        sentences: List[List[Tuple[str, int, int]]]
                        ) -> List[List[Event_whole]]:
    """Cleans and assigns event cluster.

    Args:
        event_clusters (List[Event_whole]): A list of event cluster objects
        sentence_events (List[List[Event]]): A list of sentence event lists.
        sentences (List[List[Tuple[str, int, int]]]): a list of window of sentences.
    
    Returns:
        List[List[Event_whole]]: a list of sentence event cluster lists.
    """
    sentence_events_cluster = [[] for _ in range(len(sentences))]
    # Reconstruct event_whole based on the event mentions that within the sentence(window) span
    for event_cluster in event_clusters:
        for i, sentence_event in enumerate(sentence_events):
            within_events = [eve.copy() for eve in sentence_event if (eve.event_id == event_cluster.event_id)]
            if any(within_events):
                arguments = []
                for argu in event_cluster.arguments:
                    flag = False
                    for within_event in within_events:
                        for arg in within_event.arguments:
                            if arg.mention_id.rsplit('-', 1)[0] == argu.entity_id:
                                flag = True
                                break
                        if flag:
                            break
                    if flag:
                        arguments.append(argu)
                sentence_events_cluster[i].append(Event_whole(event_cluster.event_id, event_cluster.event_type,
                                                    event_cluster.event_subtype, arguments, within_events))

    return sentence_events_cluster


def process_relation(relations: List[Relation],
                     sentence_entities: List[List[Entity]],
                     sentences: List[List[Tuple[str, int, int]]]
                    ) -> List[List[Relation]]:
    """Cleans and assigns relations

    Args:
        relations (List[Relation]): a list of Relation instances.
        sentence_entities (List[List[Entity]]): a list of sentence entity lists.
        sentences (List[List[Tuple[str, int, int]]]): a list of window of sentences.

    Returns:
        List[List[Relation]]: a list of sentence relation lists.
    """
    sentence_relations = [[] for _ in range(len(sentences))]
    for relation in relations:
        mention_id1 = relation.arg1.mention_id
        mention_id2 = relation.arg2.mention_id
        for i, entities in enumerate(sentence_entities):
            arg1_in_sent = any([mention_id1 == e.mention_id for e in entities])
            arg2_in_sent = any([mention_id2 == e.mention_id for e in entities])
            if arg1_in_sent and arg2_in_sent:
                sentence_relations[i].append(relation)
    return sentence_relations


def process_relation_entbase(relations: List[RelEntBase],
                            sentence_relations: List[List[Relation]],
                            sentences: List[List[Tuple[str, int, int]]]
                            )-> List[List[RelEntBase]]:
    sentence_relations_entbase = [[] for _ in range(len(sentences))]
    for relation in relations:
        for i, sentence_relation in enumerate(sentence_relations): # relation within the window
            # if we can find relation mentions within the span, we add this entity-based relation
            if any([rel_mention for rel_mention in sentence_relation if ((rel_mention.relation_id).rsplit('-',1)[0] == relation.relation_id)]):
                sentence_relations_entbase[i].append(relation)
            else:
                for rel_mention in sentence_relations:
                    print(rel_mention.relation_id.rsplit('-',1)[0])
                    print(relation.relation_id)
    return sentence_relations_entbase


def tokenize(sentences: List[Tuple[str, int, int]],
             entities: List[Entity],
             events: List[Event],
             language: str = 'english'
            ) -> List[Tuple[int, int, str]]:
    """Tokenizes a sentence.
    Each sentence is first split into chunks that are entity/event spans or words
    between two spans. After that, word tokenization is performed on each chunk.

    Args:
        sentences (List[Tuple[str, int, int]]): a window of sentences (tuple (text, start, end)).
        entities (List[Entity]): A list of Entity instances.
        events (List[Event]): A list of Event instances.

    Returns:
        List[Tuple[int, int, str]]: a list of token tuples. Each tuple consists
        of three elements, start offset, end offset, and token text.
    """
    all_tokens = []
    token_starts = [0]
    for sentence in sentences:
        text, start, end= sentence
        text = mask_escape(text)

        # split the sentence into chunks
        splits = {0, len(text)}
        for entity in entities:
            if entity.start >= start and entity.end <= end:
                splits.add(entity.start - start)
                splits.add(entity.end - start)
        for event in events:
            if event.trigger.start >= start and event.trigger.end <= end:
                splits.add(event.trigger.start - start)
                splits.add(event.trigger.end - start)
        splits = sorted(list(splits))
        chunks = [(splits[i], splits[i + 1], text[splits[i]:splits[i + 1]])
                for i in range(len(splits) - 1)]

        tokens = []
        if language == 'chinese':
            def _tokenize_chinese(text):
                return [c for c in jieba.cut(text) if c.strip()]

            # tokenize each chunk
            chunks = [(s, e, t, _tokenize_chinese(t))
                    for s, e, t in chunks]

            # merge chunks and add word offsets
            for chunk_start, chunk_end, chunk_text, chunk_tokens in chunks:
                last = 0
                chunk_tokens_ = []
                for token in chunk_tokens:
                    token_start = chunk_text[last:].find(token)
                    if token_start == -1:
                        raise ValueError(
                            'Cannot find token {} in {}'.format(token, text))
                    token_end = token_start + len(token)
                    chunk_tokens_.append((token_start + start + last + chunk_start,
                                        token_end + start + last + chunk_start,
                                        unmask_escape(token)))
                    last += token_end
                tokens.extend(chunk_tokens_)
        #else:
        elif language == 'arabic':
            for chunk_start, _, t in chunks:
                if language == 'english':
                    doc = nlp_en(t)
                elif language == 'arabic':
                    doc = nlp_ar(t)
                for sent in doc.sentences:
                    for tok in sent.tokens:
                        if tok.text != '':
                            tokens.append((start + chunk_start + tok.start_char,
                                        start + chunk_start + tok.end_char,
                                        unmask_escape(tok.text)))
        elif language == 'english':
            # TODO: In order to make a fair comparison with OneIE, we use NLTK's tokenizer rather than Stanza for English
            # tokenize each chunk
            chunks = [(s, e, t, wordpunct_tokenize_(t))
                    for s, e, t in chunks]

            # merge chunks and add word offsets
            tokens = []
            for chunk_start, chunk_end, chunk_text, chunk_tokens in chunks:
                last = 0
                chunk_tokens_ = []
                for token in chunk_tokens:
                    token_start = chunk_text[last:].find(token)
                    if token_start == -1:
                        raise ValueError(
                            'Cannot find token {} in {}'.format(token, text))
                    token_end = token_start + len(token)
                    chunk_tokens_.append((token_start + start + last + chunk_start,
                                        token_end + start + last + chunk_start,
                                        unmask_escape(token)))
                    last += token_end
                tokens.extend(chunk_tokens_)        

        all_tokens.extend(tokens)
        token_starts.append(len(all_tokens))
        
    return all_tokens, token_starts

def sentence2window(sentences, window_size_):
    """Convert sentences list to windows

    Args:
        sentences (List[Tuple[str, int, int]])
        window_size : int

    Returns:
        List[List[Tuple[str, int, int]]]

    """
    if window_size_ > len(sentences):
        window_size = len(sentences)
    else:
        window_size = window_size_

    output = []
    for i in range(len(sentences)-window_size+1):
        output.append(sentences[i:i+window_size])
    return output

def convert(sgm_file: str,
            apf_file: str,
            time_and_val: bool = False,
            language: str = 'english',
            window_size: int = 1) -> Document:
    """Converts a document.

    Args:
        sgm_file (str): path to a SGM file.
        apf_file (str): path to a APF file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese, chinese. Defaults to 'english'.

    Returns:
        Document: a Document instance.
    """
    sentences = read_sgm_file(sgm_file, language=language)
    doc_id, source, entities, relations, events, entity_cluster, event_cluster, relentbase =\
    read_apf_file(apf_file, time_and_val=time_and_val)
    
    if doc_id in SKIPPED_DOCS:
        return False

    # Reivse sentences
    if doc_id in DOCS_TO_REVISE_SENT:
        sentences = revise_sentences(sentences, doc_id)

    # concate windows
    sentences = sentence2window(sentences, window_size)

    # Process entities, relations, and events
    sentence_entities = process_entities(entities, sentences)
    sentence_relations = process_relation(
        relations, sentence_entities, sentences)
    sentence_events = process_events(events, sentence_entities, sentences)

    # Process entity_cluster, event_cluster, relation(entity_base)
    sentence_entity_cluster = process_entity_cluster(entity_cluster, sentences)
    sentence_relentbase = process_relation_entbase(
        relentbase, sentence_relations, sentences)
    sentence_event_cluster = process_event_cluster(event_cluster, sentence_events, sentences)    

    # Tokenization
    sentence_tokens = []
    sentence_starts = []
    for s, ent, evt in zip(sentences, sentence_entities, sentence_events):
        sent_tokens, sent_start = tokenize(s, ent, evt, language=language)
        sentence_tokens.append(sent_tokens)
        sentence_starts.append(sent_start)

    # Convert span character offsets to token indices
    sentence_objs = []
    for i, (toks, starts, ents, evts, rels, ent_cls, evt_cls, rel_ents, sent) in enumerate(zip(
            sentence_tokens, sentence_starts, sentence_entities, sentence_events,
            sentence_relations, sentence_entity_cluster, sentence_event_cluster,
            sentence_relentbase, sentences)):
        for entity in ents:
            entity.char_offsets_to_token_offsets(toks)
        for entity in ent_cls:
            for mention in entity.entity_mentions:
                mention.char_offsets_to_token_offsets(toks)
        for event in evts:
            event.trigger.char_offsets_to_token_offsets(toks)
        for event in evt_cls:
            for mention in event.event_mentions:
                mention.trigger.char_offsets_to_token_offsets(toks)
        wnd_id = '{}-{}'.format(doc_id, i)
        sentence_objs.append(Sentence(start=sent[0][1],
                                      end=sent[-1][2],
                                      text=' '.join([s[0] for s in sent]),
                                      sent_starts=starts,
                                      sent_id=wnd_id,
                                      tokens=[t for _, _, t in toks],
                                      entities=ents,
                                      relations=rels,
                                      events=evts,
                                      entity_cluster=ent_cls,
                                      relations_entbase=rel_ents,
                                      event_cluster=evt_cls))
    return Document(doc_id, sentence_objs)


def convert_batch(input_path: str,
                  output_path: str,
                  time_and_val: bool = False,
                  language: str = 'english',
                  window_size: int = 1):
    """Converts a batch of documents.

    Args:
        input_path (str): path to the input directory. Usually, it is the path 
            to the LDC2006T06/data/English or LDC2006T06/data/Chinese folder.
        output_path (str): path to the output JSON file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese, arabic. Defaults to 'english'.
    """
    if language == 'english':
        sgm_files = glob.glob(os.path.join(
            input_path, '**', 'timex2norm', '*.sgm'))
    elif language == 'chinese' or language == 'arabic':
        sgm_files = glob.glob(os.path.join(
            input_path, '**', 'adj', '*.sgm'))
    else:
        raise ValueError('Unknown language: {}'.format(language))
    print(input_path)
    print('Converting the dataset to JSON format')
    print('#SGM files: {}'.format(len(sgm_files)))
    progress = tqdm.tqdm(total=len(sgm_files))
    with open(output_path, 'w', encoding='utf-8') as w:
        for sgm_file in sgm_files:
            progress.update(1)
            apf_file = sgm_file.replace('.sgm', '.apf.xml')
            doc = convert(sgm_file, apf_file, time_and_val=time_and_val,
                          language=language, window_size=window_size)
            if doc:
                w.write(json.dumps(doc.to_dict()) + '\n')
    progress.close()


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code=ord(uchar)
            if inside_code==12288:
                inside_code=32
            elif (65373>= inside_code >= 65281):
                inside_code -= 65248
            rstring += chr(inside_code)
        if rstring == '…':
            rstring = '...'
        ss.append(rstring)
    return ''.join(ss)


def self_define_equal(current, ori_token):
    ori_str = strQ2B(ori_token)
    cur_str = ''.join([x if x!='<unk>' else '⁔' for x in current]) 
    if len(ori_str) != len(cur_str):
        return False
    for o,c in zip(ori_str, cur_str):
        if (o != c) and (c != '⁔'):
            return False
    return True


def map_decode_back_pieces(encoded_input, ori_tokens, tokenizer):
    decoded = [tokenizer.decode(x) for x in encoded_input['input_ids']]
    pieces = []
    ori_cnt = 0
    current = []
    for d in decoded:
        if d == '':
            continue
        current.append(d)
        if self_define_equal(current, ori_tokens[ori_cnt]):
            pieces.append(current)
            current = []
            ori_cnt += 1
    assert len(pieces) == len(ori_tokens)
    return pieces


def convert_to_oneie(input_path: str,
                     output_path: str,
                     language: str,
                     tokenizer: PreTrainedTokenizer):
    """Converts files to OneIE format.

    Args:
        input_path (str): path to the input file.
        output_path (str): path to the output file.
        tokenizer (PreTrainedTokenizer): wordpiece tokenizer.
    """
    print('Converting the dataset to OneIE format')
    skip_num = 0
    with open(input_path, 'r', encoding='utf-8') as r, \
            open(output_path, 'w', encoding='utf-8') as w:
        for line in r:
            doc = json.loads(line)
            for sentence in doc['sentences']:
                tokens = sentence['tokens']
                if tokenizer.__class__.__name__.startswith('XLMRoberta') and language =='chinese':
                    encoded_input = tokenizer(tokens, 
                                              is_pretokenized=True,
                                              add_special_tokens=False)
                    pieces = map_decode_back_pieces(encoded_input, tokens, tokenizer)
                else:
                    pieces = [tokenizer.tokenize(t) for t in tokens]
                token_lens = [len(x) for x in pieces]
                if 0 in token_lens:
                    skip_num += 1
                    continue
                pieces = [p for ps in pieces for p in ps]
                if len(pieces) == 0:
                    skip_num += 1
                    continue

                entity_text = {e['mention_id']: e['text']
                               for e in sentence['entities']}
                # update argument text
                for relation in sentence['relations']:
                    arg1, arg2 = relation['arg1'], relation['arg2']
                    arg1['text'] = entity_text[arg1['mention_id']]
                    arg2['text'] = entity_text[arg2['mention_id']]
                for event in sentence['events']:
                    for arg in event['arguments']:
                        arg['text'] = entity_text[arg['mention_id']]
                for event in sentence['event_cluster']:
                    for mention in event['event_mentions']:
                        for arg in mention['arguments']:
                            arg['text'] = entity_text[arg['mention_id']]

                # entities
                entities = []
                for entity in sentence['entities']:
                    assert entity['end']-entity['start'] >= 1
                    entities.append({
                        'id': entity['mention_id'],
                        'text': entity['text'],
                        'entity_type': entity['entity_type'],
                        'mention_type': entity['mention_type'],
                        'entity_subtype': entity['entity_subtype'],
                        'start': entity['start'],
                        'end': entity['end'],
                    })

                # relations
                relations = []
                for relation in sentence['relations']:
                    relations.append({
                        'id': relation['relation_id'],
                        'relation_type': relation['relation_type'],
                        'relation_subtype': '{}:{}'.format(relation['relation_type'],
                                                           relation['relation_subtype']),
                        'arguments': [
                            {
                                'entity_id': relation['arg1']['mention_id'],
                                'text': relation['arg1']['text'],
                                'role': relation['arg1']['role']
                            },
                            {
                                'entity_id': relation['arg2']['mention_id'],
                                'text': relation['arg2']['text'],
                                'role': relation['arg2']['role']
                            }
                        ]
                    })

                # events
                events = []
                for event in sentence['events']:
                    events.append({
                        'id': event['mention_id'],
                        'event_type': '{}:{}'.format(event['event_type'],
                                                     event['event_subtype']),
                        'trigger': event['trigger'],
                        'arguments': [
                            {
                                'entity_id': arg['mention_id'],
                                'text': arg['text'],
                                'role': arg['role']
                            } for arg in event['arguments']
                        ]
                    })

                # coreference
                corefs = []
                for entity in sentence['entity_cluster']:
                    if len(entity['entity_mentions']) > 1:
                        corefs.append({
                            'id': entity['entity_id'],
                            'entities': [{
                                'id': mention['mention_id'],
                                'text': mention['text'],
                                'entity_type': mention['entity_type'],
                                'mention_type': mention['mention_type'],
                                'entity_subtype': mention['entity_subtype'],
                                'start': mention['start'],
                                'end': mention['end'],
                            } for mention in entity['entity_mentions']],
                            'entity_type': entity['entity_type'],
                            'entity_subtype': entity['entity_subtype']
                        })
                
                event_corefs = []
                for event in sentence['event_cluster']:
                    if len(event['event_mentions']) > 1:
                        event_corefs.append({
                            'id': event['event_id'],
                            'events':[{
                                'id': mention['mention_id'],
                                'event_type': '{}:{}'.format(mention['event_type'],mention['event_subtype']),
                                'trigger': mention['trigger'],
                                'arguments': [
                                {
                                    'entity_id': arg['mention_id'],
                                    'text': arg['text'],
                                    'role': arg['role']
                                } for arg in mention['arguments']
                                ]    
                            } for mention in event['event_mentions']],
                            'event_type': '{}:{}'.format(mention['event_type'],mention['event_subtype']),
                            'event_arguments(entity_base)':[{
                                'entity_id': arg['entity_id'],
                                'role': arg['role']
                            } for arg in event['arguments']]
                        })

                sent_obj = {
                    'doc_id': doc['doc_id'],
                    'wnd_id': sentence['sent_id'],
                    'tokens': tokens,
                    'pieces': pieces,
                    'token_lens': token_lens,
                    'sentence': sentence['text'],
                    'entity_mentions': entities,
                    'relation_mentions': relations,
                    'event_mentions': events,
                    'entity_coreference': corefs,
                    'event_coreference': event_corefs,
                    'sentence_starts': sentence['sentence_starts'][:-1]
                }
                w.write(json.dumps(sent_obj) + '\n')
    print('skip num: {}'.format(skip_num))


def split_data(input_file: str,
               output_dir: str,
               split_path: str,
               window_size: int):
    """Splits the input file into train/dev/test sets.

    Args:
        input_file (str): path to the input file.
        output_dir (str): path to the output directory.
        split_path (str): path to the split directory that contains three files,
            train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
            files is a document ID.
    """
    print('Splitting the dataset into train/dev/test sets')
    train_docs, dev_docs, test_docs = set(), set(), set()
    # load doc ids
    with open(os.path.join(split_path, 'train.doc.txt')) as r:
        train_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'dev.doc.txt')) as r:
        dev_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'test.doc.txt')) as r:
        test_docs.update(r.read().strip('\n').split('\n'))
    
    if window_size > 10000:
        f_size = 'doc'
    else:
        f_size = 'w{}'.format(window_size)

    # split the dataset
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(os.path.join(output_dir, 'train.{}.oneie.json'.format(f_size)), 'w') as w_train, \
        open(os.path.join(output_dir, 'dev.{}.oneie.json'.format(f_size)), 'w') as w_dev, \
        open(os.path.join(output_dir, 'test.{}.oneie.json'.format(f_size)), 'w') as w_test:
        for line in r:
            inst = json.loads(line)
            doc_id = inst['doc_id']
            if doc_id in train_docs:
                w_train.write(line)
            elif doc_id in dev_docs:
                w_dev.write(line)
            elif doc_id in test_docs:
                w_test.write(line)
            else:
                print('missing!! {}'.format(doc_id))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input folder')
    parser.add_argument('-o', '--output', help='Path to the output folder')
    parser.add_argument('-s', '--split', default=None,
                        help='Path to the split folder')
    parser.add_argument('-b', '--bert',
                        help='BERT model name',
                        default='bert-large-cased')
    parser.add_argument('-c', '--bert_cache_dir',
                        help='Path to the BERT cahce directory')
    parser.add_argument('-l', '--lang', default='english',
                        help='Document language')
    parser.add_argument('--time_and_val', action='store_true',
                        help='Extracts times and values')
    parser.add_argument('-w', '--window', default=1, help='Integer for window size', type=int)

    args = parser.parse_args()
    if args.lang not in ['chinese', 'english', 'arabic']:
        raise ValueError('Unsupported language: {}'.format(args.lang))
    input_dir = os.path.join(args.input, args.lang.title())

    # Create a tokenizer based on the model name
    model_name = args.bert
    cache_dir = args.bert_cache_dir
    if model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                  cache_dir=cache_dir)
    elif model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                                     cache_dir=cache_dir)
    elif model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                                        cache_dir=cache_dir)
    elif model_name.startswith('lanwuwei'):
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                cache_dir=cache_dir, 
                                                do_lower_case=True)        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, do_lower_case=False, use_fast=False)
        #raise ValueError('Unknown model name: {}'.format(model_name))
    
    if args.window > 10000:
        f_size = 'doc'
    else:
        f_size = 'w{}'.format(args.window)
    
    with torch.no_grad():
        # Convert to doc-level JSON format
        json_path = os.path.join(args.output, '{}.{}.json'.format(args.lang, f_size))
        convert_batch(input_dir, json_path, time_and_val=args.time_and_val,
                      language=args.lang, window_size=args.window)

        # Convert to OneIE format
        oneie_path = os.path.join(args.output, '{}.{}.oneie.json'.format(args.lang, f_size))
        convert_to_oneie(json_path, oneie_path, args.lang, tokenizer=tokenizer)

        # Split the data
        if args.split:
            split_data(oneie_path, args.output, args.split, args.window)
