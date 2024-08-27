import json, logging, pickle
from mimetypes import knownfiles
from unittest.mock import NonCallableMagicMock
import torch
from torch.utils.data import Dataset
from collections import namedtuple

logger = logging.getLogger(__name__)
maven2ace = {"Bodily harm": "Injure", "Death": "Die", "Traveling": "Transport", "Getting": "Transfer-Ownership", "Receving": "Transfer-Ownership", "Commerce_buy": "Transfer-Ownership", "Giving": "Transfer-Ownership", "Submitting_documents": "Transfer-Ownership", "Supply": "Transfer-Ownership", "Commerce_sell": "Transfer-Ownership", "Renting": "Transfer-Ownership", "Exchange": "Transfer-Ownership", "Commerce_pay": "Transfer-Money", "Earnings_and_losses": "Transfer-Money", "Expensiveness": "Transfer-Money", "Protest": "Demonstrate", "Come_together": "Meet", "Social_event": "Come_together", "Communication": "Phone-Write", "Telling": "Phone-Write", "Arrest": "Arrest-Jail", "Prison": "Arrest-Jail", "Extradition": "Extradite", "Justifying": "Trial-Hearing"}
ee_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'triggers', 'roles', 'knowledge','event_types']
ee_batch_fields = ['tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'triggers', 'roles', 'wnd_ids', 'knowledge','event_types']
EEInstance = namedtuple('EEInstance', field_names=ee_instance_fields, defaults=[None] * len(ee_instance_fields))
EEBatch = namedtuple('EEBatch', field_names=ee_batch_fields, defaults=[None] * len(ee_batch_fields))

gen_batch_fields = ['input_text', 'target_text', 'enc_idxs', 'enc_attn', 'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs', 'infos']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        break_flag = False
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                break_flag = True
        if break_flag:
            continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map

def get_role_list(entities, events, id_map):
    entity_idxs = {entity['id']: (i,entity) for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(arg['entity_id'], arg['entity_id'])]
            
            # This will automatically remove multi role scenario
            if visited[i][entity_idx[0]] == 0:
                # ((trigger start, trigger end, trigger type), (argument start, argument end, role type))
                temp = ((event['trigger']['start'], event['trigger']['end'], event['event_type']),
                        (entity_idx[1]['start'], entity_idx[1]['end'], arg['role']))
                role_list.append(temp)
                visited[i][entity_idx[0]] = 1
    role_list.sort(key=lambda x: (x[0][0], x[1][0]))
    return role_list

class EEDataset(Dataset):
    def __init__(self, tokenizer, path, max_length=128, fair_compare=True):
        self.tokenizer = tokenizer
        self.path = path
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.fair_compare = fair_compare
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_length:
                continue
            self.insts.append(inst)

        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            if 'retrieval' in inst.keys():
                retrieval = inst['retrieval']
            else:
                retrieval = []
            event_types =[]
            knowledge = {}
            if len(retrieval) > 0:
                for index, item in enumerate(retrieval):
                    for et in item[1]:
                        if et in maven2ace.keys():
                            et = maven2ace[et]
                        if et in knowledge.keys():
                            knowledge[et].append(item[0])
                        else:
                            knowledge[et] = [item[0]]
                        knowledge[et] = list(set(knowledge[et]))
                    event_types.extend(item[1])
            # print(knowledge)
            # print(len(pieces))
            # print(len(knowledge_tokens))
            entities = inst['entity_mentions']
            if self.fair_compare:
                entities, entity_id_map = remove_overlap_entities(entities)
            else:
                entities = entities
                entity_id_map = {}
                
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
            assert sum(token_lens) == len(piece_idxs)
                        
            triggers = [(e['trigger']['start'], e['trigger']['end'], e['event_type']) for e in events]
            roles = get_role_list(entities, events, entity_id_map)
            
            token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]
            
            instance = EEInstance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                token_start_idxs=token_start_idxs,
                triggers = triggers,
                roles = roles,
                knowledge=knowledge,
                event_types=event_types,
            )
            self.data.append(instance)
            
        logger.info(f'Loaded {len(self)}/{len(lines)} instances from {self.path}')

    def collate_fn(self, batch):
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        piece_idxs = [inst.piece_idxs for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        token_start_idxs = [inst.token_start_idxs for inst in batch]
        triggers = [inst.triggers for inst in batch]
        roles = [inst.roles for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        knowledge = [inst.knowledge for inst in batch]
        retri_event = [inst.event_types for inst in batch]
        return EEBatch(
            tokens=tokens,
            pieces=pieces,
            piece_idxs=piece_idxs,
            token_lens=token_lens,
            token_start_idxs=token_start_idxs,
            triggers=triggers,
            roles=roles,
            wnd_ids=wnd_ids,
            knowledge=knowledge,
            event_types=retri_event,
        )

class GenDataset(Dataset):
    def __init__(self, tokenizer, max_length, path, max_output_length=None, unseen_types=[], no_bos=False):
        self.tokenizer = tokenizer
        self.max_length = self.max_output_length = max_length
        if max_output_length is not None:
            self.max_output_length = max_output_length
        self.path = path
        self.no_bos = no_bos # if you use bart, then this should be False; if you use t5, then this should be True
        self.data = []
        self.load_data(unseen_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, unseen_types):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        for l_in, l_out, l_info in zip(data['input'], data['target'], data['all']):
            if len(unseen_types) > 0:
                if isinstance(l_info, tuple):
                    # instance base
                    if l_info[1] in unseen_types:
                        continue
                else:
                    # trigger base, used in argument model
                    if l_info['event type'] in unseen_types:
                        continue
            self.data.append({
                'input': l_in,
                'target': l_out,
                'info': l_info
            })
        logger.info(f'Loaded {len(self)} instances from {self.path}')

    def collate_fn(self, batch):
        input_text = [x['input'] for x in batch]
        target_text = [x['target'] for x in batch]
        
        # encoder inputs
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.max_length)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(target_text, return_tensors='pt', padding=True, max_length=self.max_output_length)
        dec_idxs = targets['input_ids']
        # print(self.tokenizer.decode(dec_idxs[0]))
        batch_size = dec_idxs.size(0)
        dec_idxs[:, 0] = self.tokenizer.eos_token_id
        dec_attn = targets['attention_mask']
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return GenBatch(
            input_text=input_text,
            target_text=target_text,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            raw_lbl_idxs=raw_lbl_idxs,
            infos=[x['info'] for x in batch]
        )
