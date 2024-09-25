import sys
sys.path.append('./')
import random
random.seed(42)

from convert.sampler import get_positive_type_role

def match_sublist(the_list, to_match):
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match)]
    return matched_list


def rel_sort(text, rels):
    new_rels = []
    rels_mapper = {}
    for rel in rels:
        if rel['relation'] not in rels_mapper:
            type_id = random.randint(0, 1000)
            rels_mapper[rel['relation']] = type_id
        else:
            type_id = rels_mapper[rel['relation']] 
        head_offset = match_sublist(list(text), list(rel['head']))
        tail_offset = match_sublist(list(text), list(rel['tail']))
        if len(head_offset) == 0:
            head_offset = [[-100,-100],]
        if len(tail_offset) == 0:
            tail_offset = [[-100,-100],]
        head_offset = head_offset[0]
        tail_offset = tail_offset[0]
        new_rels.append([rel, [head_offset[0], type_id, tail_offset[0]]])
    new_rels = sorted(new_rels, key=lambda x: (x[1][1], x[1][0], x[1][2]))
    new_rels = [it[0] for it in new_rels]
    rels_list = sorted(rels_mapper.items(), key=lambda x: x[1])
    rels_list = [it[0] for it in rels_list]
    return new_rels, rels_list


def ee_sort(text, events):
    new_events = []
    type_mapper = {}
    role_mapper = {}
    positive_type, positive_role, positive_type_role = get_positive_type_role(events, "EE")
    for ttype in positive_type:
        type_id = random.randint(0, 1000)
        type_mapper[ttype] = type_id
    for rrole in positive_role:
        role_id = random.randint(0, 1000)
        role_mapper[rrole] = role_id

    for event in events:
        type_id = type_mapper[event['event_type']]
        event_offset = match_sublist(list(text), list(event['event_trigger']))
        if len(event_offset) == 0:
            event_offset = [[-100,-100],]
        event_offset = event_offset[0]
        new_args = []
        for arg in event['arguments']:
            role_id = role_mapper[arg['role']]
            arg_offset = match_sublist(list(text), list(arg['argument']))
            if len(arg_offset) == 0:
                arg_offset = [[-100,-100],]
            arg_offset = arg_offset[0]
            new_args.append([arg, [arg_offset[0], role_id]])
        new_args = sorted(new_args, key=lambda x: (x[1][1], x[1][0]))
        new_args = [it[0] for it in new_args]
        event['arguments'] = new_args
        new_events.append([event, [event_offset[0], type_id]])
    new_events = sorted(new_events, key=lambda x: (x[1][1], x[1][0]))
    new_events = [it[0] for it in new_events]
    new_type_role_dict = {}
    for key, value in positive_type_role.items():
        sorted_value = sorted(value, key=lambda x: role_mapper[x])
        new_type_role_dict[key] = sorted_value
    new_type_role_dict = dict(sorted(new_type_role_dict.items(), key=lambda x: type_mapper[x[0]]))
    return new_events, new_type_role_dict



def ent_sort(text, ents):
    new_ents = []
    ents_mapper = {}
    for ent in ents:
        if ent['entity_type'] not in ents_mapper:
            type_id = random.randint(0, 1000)
            ents_mapper[ent['entity_type']] = type_id
        else:
            type_id = ents_mapper[ent['entity_type']] 
        ent_offset = match_sublist(list(text), list(ent['entity']))
        if len(ent_offset) == 0:
            ent_offset = [[-100,-100],]
        ent_offset = ent_offset[0]
        new_ents.append([ent, [ent_offset[0], type_id]])
    new_ents = sorted(new_ents, key=lambda x: (x[1][1], x[1][0]))
    new_ents = [it[0] for it in new_ents]
    ents_list = sorted(ents_mapper.items(), key=lambda x: x[1])
    ents_list = [it[0] for it in ents_list]
    return new_ents, ents_list



def eet_sort(text, ents):
    new_ents = []
    ents_mapper = {}
    for ent in ents:
        if ent['event_type'] not in ents_mapper:
            type_id = random.randint(0, 1000)
            ents_mapper[ent['event_type']] = type_id
        else:
            type_id = ents_mapper[ent['event_type']] 
        ent_offset = match_sublist(list(text), list(ent['event_trigger']))
        if len(ent_offset) == 0:
            ent_offset = [[-100,-100],]
        ent_offset = ent_offset[0]
        new_ents.append([ent, [ent_offset[0], type_id]])
    new_ents = sorted(new_ents, key=lambda x: (x[1][1], x[1][0]))
    new_ents = [it[0] for it in new_ents]
    ents_list = sorted(ents_mapper.items(), key=lambda x: x[1])
    ents_list = [it[0] for it in ents_list]
    return new_ents, ents_list
