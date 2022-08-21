#!/usr/bin/env python

"""
This code implements rule-based event detection.
"""

from annotation import read_dataset, write_dataset, get_head_word_index, EventNugget
from timeit import default_timer as timer
from subprocess import Popen, PIPE
from nltk.corpus import wordnet as wn
from pattern.en import conjugate
from polyglot.text import Word
import string
import nltk
import sys
import gzip
import argparse
import pdb


PREFIX_LIST = [
    # Negative prefixes
    'un', 'in', 'im', 'il', 'ir', 'non', 'mis', 'mal', 'dis', 'anti', 'de',
    'under', 'semi', 'mini', 'ex', 'sub', 'infra',
    # Positive prefixes
    're', 'over', 'equi', 'micro', 'macro', 'mega', 'extra', 'prime', 'post',
    'retro', 'bi', 'multi', 'pro', 'auto', 'co', 'con',
    # Neutral prefixes
    'inter', 'super', 'super','peri', 'ante', 'pre', 'semi', 'mono', 'tri',
    'quad', 'penta', 'hex', 'sept', 'oct', 'dec'
]
PREFIXES = {p:True for p in PREFIX_LIST}

# Non-eventive verb senses
NONEV_VERB_SENSE_LIST = [
    'mean.v.03', 'equal.v.01', 'look.v.02', 'appear.v.04',
    'seem.v.03', 'seem.v.04'
]
NONEV_VERB_SENSES = {s:True for s in NONEV_VERB_SENSE_LIST}

# Verbs which may take a complement to form a single event nugget
XCOMP_VERB_LIST = [
    'become', 'turn', 'smell', 'make', 'get', 'find', 'consider'
]
XCOMP_VERBS = {v:True for v in XCOMP_VERB_LIST}

# Verbs which may take a direct object to form a single event nugget
DOBJ_VERBS = {
    'catch' : {'measles':True},
    'charge' : {'tax':True},
    'create' : {'heat':True},
    'die' : {'death':True},
    'do' : {'research':True, 'test':True},
    'feel' : {'need':True},
    'find' : {'solution':True},
    'form' : {'bond':True},
    'fulfill' : {'requirement':True},
    'gain' : {'right':True},
    'get' : {'blister':True, 'chickenpox':True, 'control':True, 'disease':True,
             'measles':True, 'order':True, 'right':True, 'stroke':True},
    'give' : {'clearance':True, 'explanation':True, 'income':True, 'report':True,
              'response':True, 'right':True, 'vaccine':True},
    'have' : {'cancer':True, 'chickenpox':True, 'condition':True, 'disease':True,
              'dyslexia':True, 'effect':True, 'idea':True, 'incident':True,
              'job':True, 'measles':True, 'memory':True, 'meningitis':True,
              'money':True, 'need':True, 'problem':True, 'rash':True, 'requirement':True,
              'right':True, 'sign':True, 'symptom':True, 'visitor':True},
    'impose' : {'tax':True},
    'leave' : {'scar':True},
    'lose' : {'job':True},
    'make' : {'bond':True, 'change':True, 'coffee':True, 'copy':True, 'law':True,
              'money':True, 'profit':True, 'state':True, 'tea':True, 'urine':True},
    'mark' : {'end':True},
    'pass' : {'law':True},
    'play' : {'role':True},
    'produce' : {'response':True},
    'punch' : {'hole':True},
    'receive' : {'support':True},
    'solve' : {'problem':True},
    'take' : {'control':True, 'delivery':True, 'medicine':True, 'picture':True,
              'place':True},
    'write' : {'problem':True},
}


def get_morphemes(word):
    """
    Returns a list of morphemes of the given word.
    """
    return ' '.join([str(m) for m in Word(word, language='en').morphemes])


def detect_events_in_sentence(sent):
    token2phrase = {}
    for phrase in sent.phrases:
        phrase_tokens = []
        for token_id in phrase.tokens:
            token = sent.tokens[token_id-1]
            phrase_tokens.append(token)
            token2phrase[token] = phrase
        phrase.tokens = phrase_tokens

        head_word_idx = get_head_word_index(phrase.tokens)
        if head_word_idx is None:
            continue
        phrase.head_word = phrase.tokens[head_word_idx]
        phrase.sense_id = phrase.head_word.sense_id

        if phrase.tokens[0].lemma is None:
            continue
        first_lemma = phrase.tokens[0].lemma.lower()
        if first_lemma == 'be' or first_lemma == 'the':
            # Avoid a verb phrase starting with a be-verb or
            # a noun phrase staring with 'the'
            continue

        if phrase.head_word.pos is None:
            continue

        # Verb phrases
        if is_main_verb(phrase.head_word):
            tokens = remove_determiners(phrase)
            create_event_nugget(tokens, sent)
            continue

        # Noun phrases
        if is_noun(phrase.head_word):
            if is_eventive_noun_phrase(phrase):
                tokens = remove_determiners(phrase)
                create_event_nugget(tokens, sent)
            continue

    skip_tokens = {}
    for token in sent.tokens:
        if token in skip_tokens:
            continue
        if token in token2phrase:
            continue
        if token.lemma is None or token.pos is None:
            continue

        # Verbs
        if is_main_verb(token):
            if is_noneventive_verb(token):
                continue
            comps = xcomp_verb_phrase(token)
            if comps:
                for comp in comps:  # skip complements in verb phrases
                    skip_tokens[comp] = True
                continue
            dobjs = dobj_verb_phrase(token)
            if dobjs:
                for dobj in dobjs:  # skip direct objects in verb phrases
                    skip_tokens[dobj] = True
                continue

            # Adverbs can be a part of an event nugget based on verbs
            adverb = get_modifying_adverb(token)
            if adverb is None or adverb in token2phrase is not None or \
               not is_eventive_adverb(adverb):  # single-word verbs
                create_event_nugget([token], sent)
            else:  # verb + adverb
                if adverb.token_id < token.token_id:  # preceding
                    create_event_nugget([adverb, token], sent)
                else:
                    create_event_nugget([token, adverb], sent)
            continue

        # Nouns
        if is_noun(token):
            if is_eventive_noun(token):
                create_event_nugget([token], sent)
            continue

        # Adjectives
        if is_adjective(token):
            if is_eventive_adjective(token):
                create_event_nugget([token], sent)
            continue

    token2phrase.clear()
    skip_tokens.clear()


def detect_events(dataset, log_interval=None):
    start = timer()
    for i, doc in enumerate(dataset.docs):
        for sent in doc.sents:
            detect_events_in_sentence(sent)
        if log_interval is not None and (i+1) % log_interval == 0:
            print(f"Processed {i+1} documents.  Elapsed time: {timer()-start:.3f} [s]")
            sys.stdout.flush()


def create_event_nugget(tokens, sent):
    """
    Creates an event nugget.
    """
    e = EventNugget()
    for token in tokens:
        e.tokens.append(token.token_id)
    e.sent = sent
    sent.event_nuggets.append(e)


def remove_determiners(phrase):
    non_dt_tokens = [t for t in phrase.tokens if t.pos != 'DT']

    # Found no determiner
    if len(non_dt_tokens) == len(phrase.tokens):
        return phrase.tokens

    return non_dt_tokens


def lookup_sense(sense_key):
    try:
        synset = wn.lemma_from_key(sense_key).synset().name()
    except:
        synset = None
    return synset


def is_main_verb(token):
    """
    Returns true if the given token is a main verb.
    """
    if token.pos.startswith('VB'):  # verbs
        # Simple rule: all main verbs except be-verbs and auxiliary verbs
        if token.lemma is None:
            return False
        if token.lemma.lower() == 'be':  # be-verbs
            return False
        if token.lemma.lower() == 'have' or token.lemma.lower() == 'need':
            token_idx = token.sent.tokens.index(token)
            if token_idx < len(token.sent.tokens) - 1 and \
               token.sent.tokens[token_idx+1].lemma.lower() == 'to':
                return False  # `have to', `need to'
        for head_dep in token.head_deps:
            if head_dep.rel == 'aux':  # auxiliary verbs
                return False
        return True
    return False


def is_noneventive_verb(verb):
    if verb.sense_id is None:
        return False
    synset = lookup_sense(verb.sense_id)
    if synset is None:
        return False
    return synset in NONEV_VERB_SENSES


def xcomp_verb_phrase(verb):
    """
    Returns complements if the given verb forms a phrasal event nugget with complements.
    Examples are 'get sick', 'smell good', 'make ... clear', etc.

    Output:
    comps (list) :: a list of complements
    """
    if verb.sense_id is None:
        return []
    if verb.lemma is None:
        return []
    if verb.lemma.lower() not in XCOMP_VERBS:
        return []

    comps = []
    for tail_dep in verb.tail_deps:
        tail = tail_dep.tail
        if tail_dep.rel != 'xcomp':
            continue
        if verb.token_id >= tail:
            continue
        tail = verb.sent.tokens[tail-1]
        if not is_adjective(tail) and not is_adverb(tail) and tail.pos != 'VBN':
            continue

        create_event_nugget([verb, tail], verb.sent)
        comps.append(tail)
    return comps


def dobj_verb_phrase(verb):
    """
    Returns direct objects if the given verb forms a phrasal event nugget with direct objects.
    Examples are 'make (a) profit',  'have (an) effect, etc.

    Output:
    dobjs (list) :: a list of direct objects
    """
    if verb.sense_id is None:
        return []
    if verb.lemma is None:
        return []
    if verb.lemma.lower() not in DOBJ_VERBS:
        return []

    dobjs = []
    for tail_dep in verb.tail_deps:
        tail = tail_dep.tail
        if tail_dep.rel != 'dobj':
            continue
        if verb.token_id >= tail:
            continue
        tail = verb.sent.tokens[tail-1]
        if not is_noun(tail):
            continue
        if tail.lemma is None:
            continue
        if tail.lemma.lower() not in DOBJ_VERBS[verb.lemma.lower()]:
            continue

        create_event_nugget([verb, tail], verb.sent)
        dobjs.append(tail)
    return dobjs


def get_modifying_adverb(verb):
    """
    Returns an adverb modifying the given verb.
    """
    for tail_dep in verb.tail_deps:
        if tail_dep.rel == 'advmod':
            tail = verb.sent.tokens[tail_dep.tail-1]
            if is_adverb(tail):
                return tail
    return None


def is_noun(token):
    """
    Returns true if the given token is a noun.
    """
    return token.pos.startswith('NN')


def is_eventive_noun(noun):
    """
    Returns true if the given token is an eventive noun.  We check if the
    given noun's sense is in eventive senses.
    """
    if noun.sense_id is None:
        return False
    for head_dep in noun.head_deps:
        if head_dep.rel == 'compound':
            head = noun.sent.tokens[head_dep.head-1]
            if is_noun(head):
                return False  # a non-head of a compound noun cannot be eventive
    synset = lookup_sense(noun.sense_id)
    if synset is None:
        return False
    return synset in eventive_synsets


def is_eventive_noun_phrase(np):
    """
    Returns true if the given noun phrase is eventive.  We check if the
    given phrase's sense is in eventive senses.
    """
    if np.sense_id is None:
        return False
    synset = lookup_sense(np.sense_id)
    if synset is None:
        return False
    return synset in eventive_synsets


def is_eventive_wikipedia_concept(wc):
    """
    Returns true if the given Wikipedia concept is eventive.  We check if the
    sense of the head word of the concept's gloss is in eventive senses.
    """
    if wc.gloss_head_sense is None:
        return False
    synset = lookup_sense(wc.gloss_head_sense)
    if synset is None:
        return False
    return synset in eventive_synsets


def is_adjective(token):
    """
    Returns true if the given token is an adjective.
    """
    return token.pos.startswith('JJ')  # adjectives


def is_eventive_adjective(token):
    """
    Returns true if the given token is an eventive adjective.  We use simple
    and conservative heuristics that detects adjectives originated from present
    or past participles of verbs as events, e.g., `sparkling' and 'man-made'.
    """
    word = token.text.lower()
    if '-' in word:  # e.g., 'well-known'
        # Finding a head subword is difficult.  First, we check if
        # there is a verb in subwords.  If not, use the last token.
        verb = None
        for subword, pos in reversed(nltk.pos_tag(word.split('-'))):
            if pos.startswith('VB'):
                verb = subword  # found a verb
                break
        word = verb if verb is not None else word.split('-')[-1]

    # Remove a prefix, if any, to make verb conjugation and checking easier below.
    morphemes = get_morphemes(word)
    if morphemes and morphemes[0] in PREFIXES:
        word = word[len(morphemes[0]):]
    if len(word) == 0:
        return False

    conj_inf = conjugate(word, 'inf')
    if wn.lemmas(conj_inf, wn.VERB):
        if conjugate(conj_inf, 'ppart') == word:
            return True
        if conjugate(conj_inf, 'part') == word:
            return True

    return False


def is_adverb(token):
    """
    Returns true if the given token is an adverb.
    """
    return token.pos.startswith('RB')  # adverbs


def is_eventive_adverb(token):
    """
    Returns true if the given token is an eventive adverb.  We use simple
    and conservative heuristics that detects adverbs originated from present
    or past participles of verbs as events, e.g., `sparklingly' and
    `unexpectedly'.
    """
    word = token.text.lower()

    # Remove a prefix, if any, to make verb conjugation and checking easier below.
    morphemes = get_morphemes(word)
    if morphemes and morphemes[0] in PREFIXES:
        word = word[len(morphemes[0]):]
    if len(word) == 0:
        return False

    # Pertainyms are relational adjectives
    lm_adjs = [lm_ptn for lm in wn.lemmas(word, pos=wn.ADV)
               for lm_ptn in lm.pertainyms()]
    if word.endswith('ly'):
        lm_adjs.extend(wn.lemmas(word[:-2]))
    words = set([lm_adj.name() for lm_adj in lm_adjs])
    for word in words:
        conj_inf = conjugate(word, 'inf')
        if wn.lemmas(conj_inf, wn.VERB):
            if conjugate(conj_inf, 'ppart') == word:
                return True
            if conjugate(conj_inf, 'part') == word:
                return True

    return False


def read_eventive_synsets(input_file):
    if input_file.endswith('.gz'):
        fin = gzip.open(input_file, 'rt')
    else:
        fin = open(input_file, 'r')

    eventive_synsets = {}
    for line in fin:
        _, synset, _ = line.rstrip('\n').split('\t')
        eventive_synsets[synset] = True

    fin.close()
    return eventive_synsets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, metavar='str', default='out/sw100_wsd_phrase.json.gz',
                        help="input file (gzipped JSON)")
    parser.add_argument('--output', type=str, metavar='str', default='out/sw100_wsd_phrase_rule.json.gz',
                        help="output file (gzipped JSON)")
    parser.add_argument('--event-synsets', type=str, metavar='str', default='resources/wordnet_gloss_events.txt.gz',
                        help="file for eventive noun synsets")
    args = parser.parse_args()

    dataset = read_dataset(args.input)
    print(f"Loaded input data ({len(dataset.docs)} documents) from {args.input}")

    eventive_synsets = read_eventive_synsets(args.event_synsets)
    print(f"Loaded {len(eventive_synsets)} eventive noun synsets from {args.event_synsets}")

    detect_events(dataset)
    print("Finished event detection.")

    write_dataset(dataset, args.output)
    print(f"Wrote output to {args.output}")
