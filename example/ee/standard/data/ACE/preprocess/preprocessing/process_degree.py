import json
from argparse import ArgumentParser
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer


def map_index(pieces):
    idxs = []
    for i, piece in enumerate(pieces):
        if i == 0:
            idxs.append([0, len(piece)])
        else:
            _, last = idxs[-1]
            idxs.append([last, last + len(piece)])
    return idxs

def map_decode_back_pieces(encoded_input, ori_tokens, tokenizer):
    decoded = [tokenizer.decode(x) for x in encoded_input['input_ids']]
    pieces = []
    ori_cnt = 0
    current = []
    for d in decoded:
        if d == '':
            continue
        current.append(d)
        if ''.join(current) == ori_tokens[ori_cnt]:
            pieces.append(current)
            current = []
            ori_cnt += 1
    assert len(pieces) == len(ori_tokens)
    return pieces

def convert(input_file, output_file, tokenizer, window_size_=3):
    with open(input_file, 'r', encoding='utf-8') as r, \
            open(output_file, 'w', encoding='utf-8') as w:
        for line in r:
            doc = json.loads(line)
            doc_id = doc['doc_key']
            sentences = doc['sentences']
            sent_num = len(sentences)
            total_tokens = sum([len(sent) for sent in sentences])
            coref_entities = doc['clusters']
            coref_events = doc['event_clusters']
            # upper bound on token index for checking index in range
            sent_starts = doc['_sentence_start'] + [total_tokens]
            entities = doc.get('ner', [[] for _ in range(sent_num)])
            relations = doc.get('relations', [[] for _ in range(sent_num)])
            events = doc.get('events', [[] for _ in range(sent_num)])

            if window_size_ > sent_num:
                window_size = sent_num
            else:
                window_size = window_size_
                
            offset = 0
            for i in range(sent_num - window_size + 1):
                wnd_sent_starts = sent_starts[i:i+window_size+1]
                wnd_start, wnd_end = wnd_sent_starts[0], wnd_sent_starts[-1]

                def slice_fn(lst, ind, wnd): 
                    return [item for j in range(wnd) for item in lst[ind+j]]
                wnd_tokens, wnd_entities, wnd_relations, wnd_events = [slice_fn(
                    lst, i, window_size) for lst in [sentences, entities, relations, events]]

                wnd_id = '{}-{}'.format(doc_id, i)
                pieces = [tokenizer.tokenize(t) for t in wnd_tokens]
                word_lens = [len(p) for p in pieces]            

                wnd_entities_ = []
                wnd_entity_map = {}
                for j, (start, end, entity_type) in enumerate(wnd_entities):
                    start, end = start - offset, end - offset + 1
                    entity_id = '{}-E{}'.format(wnd_id, j)
                    entity = {
                        'id': entity_id,
                        'start': start, 'end': end,
                        'entity_type': entity_type,
                        # Mention types are not included in DyGIE++'s format
                        'mention_type': 'UNK',
                        'text': ' '.join(wnd_tokens[start:end])}
                    wnd_entities_.append(entity)
                    wnd_entity_map[(start, end)] = entity

                wnd_relations_ = []
                for j, (start1, end1, start2, end2, rel_type) in enumerate(wnd_relations):
                    start1, end1 = start1 - offset, end1 - offset + 1
                    start2, end2 = start2 - offset, end2 - offset + 1
                    arg1 = wnd_entity_map[(start1, end1)]
                    arg2 = wnd_entity_map[(start2, end2)]
                    relation_id = '{}-R{}'.format(wnd_id, j)
                    rel_type = rel_type.split('.')[0]
                    relation = {
                        'relation_type': rel_type,
                        'id': relation_id,
                        'arguments': [
                            {
                                'entity_id': arg1['id'],
                                'text': arg1['text'],
                                'role': 'Arg-1'
                            },
                            {
                                'entity_id': arg2['id'],
                                'text': arg2['text'],
                                'role': 'Arg-2'
                            },
                        ]
                    }
                    wnd_relations_.append(relation)
                
                # parse coref entities
                # for each entity mention in a coref, only look up the obj in the dict if they are in the window
                wnd_coref_ents = [[wnd_entity_map[(ent[0]-offset, ent[1]-offset+1)] for ent in coref if wnd_start <= ent[0]
                                   and ent[1] < wnd_end and (ent[0]-offset, ent[1]-offset+1) in wnd_entity_map] for coref in coref_entities]
                wnd_coref_ents_ = []
                for j, ent_list in enumerate(wnd_coref_ents):
                    if len(ent_list) > 1:
                        wnd_coref_ents_.append({
                            'id': '{}-CE{}'.format(wnd_id, j),
                            'entities': ent_list
                        })

                wnd_events_ = []
                wnd_event_map = {}
                for j, event in enumerate(wnd_events):
                    event_id = '{}-EV{}'.format(wnd_id, j)
                    if len(event[0]) == 3:
                        trigger_start, trigger_end, event_type = event[0]
                    elif len(event[0]) == 2:
                        trigger_start, event_type = event[0]
                        trigger_end = trigger_start
                    trigger_start, trigger_end = trigger_start - offset, trigger_end - offset + 1
                    event_type = event_type.replace('.', ':')
                    args = event[1:]
                    args_ = []
                    for arg_start, arg_end, role in args:
                        arg_start, arg_end = arg_start - offset, arg_end - offset +1
                        arg = wnd_entity_map[(arg_start, arg_end)]
                        args_.append({
                            'entity_id': arg['id'],
                            'text': arg['text'],
                            'role': role
                        })
                    event_obj = {
                        'event_type': event_type,
                        'id': event_id,
                        'trigger': {
                            'start': trigger_start,
                            'end': trigger_end,
                            'text': ' '.join(wnd_tokens[trigger_start:trigger_end])
                        },
                        'arguments': args_
                    }
                    wnd_events_.append(event_obj)
                    wnd_event_map[(trigger_start, trigger_end)] = event_obj

                # parse coref events
                wnd_coref_evts = [[wnd_event_map[(evt[0]-offset, evt[1]-offset+1)]
                                   for evt in coref if wnd_start <= evt[0] and evt[1] < wnd_end 
                                   and (evt[0]-offset, evt[1]-offset+1) in wnd_event_map] for coref in coref_events]
                wnd_coref_evts_ = []
                for j, evt_list in enumerate(wnd_coref_evts):
                    if len(evt_list) > 1:
                        wnd_coref_evts_.append({
                            'id': '{}-CEV{}'.format(wnd_id, j),
                            'events': evt_list
                        })

                wnd_ = {
                    'doc_id': doc_id,
                    'wnd_id': wnd_id,
                    'entity_mentions': wnd_entities_,
                    'relation_mentions': wnd_relations_,
                    'event_mentions': wnd_events_,
                    'entity_coreference': wnd_coref_ents_,
                    'event_coreference': wnd_coref_evts_,
                    'tokens': wnd_tokens,
                    'pieces': [p for w in pieces for p in w],
                    'token_lens': word_lens,
                    'sentence': ' '.join(wnd_tokens),
                    'sentence_starts': [x-offset for x in wnd_sent_starts[:-1]],                  
                }
                w.write(json.dumps(wnd_) + '\n')
                offset += len(sentences[i])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-b', '--bert', help='BERT model name', default='bert-large-cased')
    parser.add_argument('-w', '--window', default=1, help='Integer for window size', type=int)
    args = parser.parse_args()
    model_name = args.bert
    if model_name.startswith('bert-'):
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    elif model_name.startswith('roberta-'):
        bert_tokenizer = RobertaTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, do_lower_case=False, use_fast=False)
    
    convert(args.input, args.output, bert_tokenizer, args.window)
