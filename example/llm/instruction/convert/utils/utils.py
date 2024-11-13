import json
import hashlib

def get_string_list(l):
    return "[" + ', '.join(l) + "]"


def get_string_dict(d):
    s_d = []
    for k, value in d.items():
        s_value =  k + ": " + "[" + ', '.join(value) + "]"
        s_d.append(s_value)
    return '{' + ', '.join(s_d) + '}'


def write_to_json(path, datas):
    with open(path, 'w', encoding='utf-8') as writer:
        for data in datas:
            writer.write(json.dumps(data, ensure_ascii=False)+"\n")


def stable_hash(input_str):
    sha256 = hashlib.sha256()
    sha256.update(input_str.encode('utf-8'))
    return sha256.hexdigest()


def match_sublist(the_list, to_match):
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match)]
    return matched_list


def spotext2schema(text):
    spt = text.split('_')
    return spt[0], '_'.join(spt[1:-1]), spt[-1]


def schema2spotext(head_type, relation, tail_type):
    return f"{head_type}_{relation}_{tail_type}"