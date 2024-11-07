import openai
import json
import random
from tqdm import tqdm
import argparse
import os

entity_types = {
    "tacrev": ['URL', 'LOCATION', 'IDEOLOGY', 'CRIMINAL CHARGE', 'TITLE', 'STATE OR PROVINCE', 'DATE', 'PERSON', 'NUMBER', 'CITY', 'DURATION', 'CAUSE OF DEATH', 'COUNTRY', 'NATIONALITY', 'RELIGION', 'ORGANIZATION', 'MISCELLANEOUS'],
    # "SciERC": ['Generic', 'Material', 'Method', 'Metric', 'OtherScientificTerm', 'Task'],
    "retacred": ['IDEOLOGY', 'ORGANIZATION', 'URL', 'PERSON', 'DURATION', 'COUNTRY', 'LOCATION', 'NATIONALITY', 'TITLE', 'RELIGION', 'NUMBER', 'CITY', 'CAUSE OF DEATH', 'DATE', 'STATE OR PROVINCE', 'CRIMINAL CHARGE'],
    "tacred": ['COUNTRY', 'IDEOLOGY', 'LOCATION', 'DATE', 'PERSON', 'NATIONALITY', 'RELIGION', 'CITY', 'MISCELLANEOUS', 'CAUSE OF DEATH', 'TITLE', 'URL', 'NUMBER', 'ORGANIZATION', 'STATE OR PROVINCE', 'DURATION', 'CRIMINAL CHARGE']
}

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-ak', type=str, required=True)
    parser.add_argument('--demo_path', '-dp', type=str, required=True, help="The directory of demonstration data.")
    parser.add_argument('--output_dir', type=str, required=True, help="The output directory of generated data.")
    parser.add_argument('--dataset', type=str, required=True, choices=["tacred", "tacrev", "retacred"])
    parser.add_argument('--k', type=int, default=3, help="k-shot demonstrations")
    args = parser.parse_args()
    
    openai.api_key = args.api_key

    input_file = args.demo_path
    datasetname = args.dataset
    output_file = os.path.join(args.output_dir, "generated.json")
    data = []
    label_list = {}
    with open(input_file,'r') as f:
        data = json.load(f)
    random.shuffle(data)
    for line in data:
        rel = line['relation']
        if rel not in label_list:
            label_list[rel] = [line]
        else:
            label_list[rel].append(line)


    '''
    One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. 
    The head entity has the relation with the tail entity and entities are pre-categorized as the following types: URL, LOCATION, IDEOLOGY, CRIMINAL CHARGE, TITLE, STATE OR PROVINCE, DATE, PERSON, NUMBER, CITY, DURATION, CAUSE OF DEATH, COUNTRY, NATIONALITY, RELIGION, ORGANIZATION, MISCELLANEOUS. 
    Here are some samples for relation 'org:founded_by':
    Relation: org:founded_by. Context: President Lee Teng-hui confers the Order of the Brilliant Star with a Violet Grand Cordon on Samuel Noordhoff , founder of the Noordhoff Craniofacial Foundation , for his devoted service to local citizens over the past four decades. Head Entity: Noordhoff Craniofacial Foundation . Head Type: ORGANIZATION. Tail Entity: Samuel Noordhoff. Tail Type: PERSON.
    Relation: org:founded_by. Context: Talansky is also the US contact for the New Jerusalem Foundation , an organization founded by Olmert while he was Jerusalem 's mayor . Head Entity: New Jerusalem Foundation. Head Type: ORGANIZATION. Tail Entity: Olmert. Tail Type: PERSON.
    Relation: org:founded_by. Context: Sharpton has said he will not endorse any candidate until hearing more about their views on civil rights and other issues at his National Action Network convention next week in New York City . Head Entity: National Action Network. Head Type: ORGANIZATION. Tail Entity: his. Tail Type: PERSON.
    Relation: org:founded_by. Context: `` We believe that we can best serve our clients by offering a single multistrategy hedge fund platform , '' wrote John Havens , who was a founder of Old Lane with Pandit and is president of the alternative investment group . Head Entity: Old Lane. Head Type: ORGANIZATION. Tail Entity: John Havens. Tail Type: PERSON.
    Generate more samples for the relation 'org:founded_by'.
    '''

    with open(output_file,'a') as f:
        for k,v in tqdm(label_list.items()):
            prompt = "One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. The head entity has the relation with the tail entity and entities are pre-categorized as the following types: " + \
            (', '.join(entity_types[datasetname])) + \
            ". Here are some samples for relation '" + k + "':\n"
            for i in range(args.k):
                sample = "Relation: " + k + ". Context: " + ' '.join([convert_token(token) for token in v[i]['token']]) + ' ' + "Head Entity: " + ' '.join([convert_token(token) for token in v[i]['token'][v[i]['subj_start']:v[i]['subj_end']+1]]) + '. ' + "Head Type: " + v[i]['subj_type'] + '. ' + "Tail Entity: " + ' '.join([convert_token(token) for token in v[i]['token'][v[i]['obj_start']:v[i]['obj_end']+1]]) + ". " + "Tail Type: " + v[i]['obj_type'] + ".\n"
                prompt = prompt + sample
            prompt = prompt + "Generate more samples like above for the relation '" + k + "'."
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt = prompt,
                temperature=1,
                max_tokens=3500
            )
            res = response['choices'][0]['text'].split('\n')
            for line in res:
                if len(line) == 0:
                    continue
                try:
                    DAdata = {}
                    data1 = line.split('Relation:')[-1].strip()
                    onepoint = data1.index('.')
                    relation = data1[:onepoint]
                    if relation == k:
                        relation = k
                    else:
                        continue

                    # text
                    data2 = data1.split('Context:')[-1].strip()
                    data2lower = data2.lower()
                    if "head entity:" in data2lower:
                        textend = data2lower.index('head entity:')
                        text = data2[:textend].strip()
                        data3 = data2[textend+len('head entity:'):].strip()
                    else:
                        continue

                    DAdata['text'] = text

                    # head entity
                    data3lower = data3.lower()
                    if ". head type:" in data3lower:
                        headend = data3lower.index(". head type:")
                        head = data3[:headend]
                        data4 = data3[headend + len(". head type:"):].strip()
                    else:
                        continue
                    
                    # head type
                    data4lower = data4.lower()
                    if ". tail entity:" in data4lower:
                        htend = data4lower.index(". tail entity:")
                        headtype = data4[:htend]
                        if headtype in entity_types[datasetname] or headtype.replace('_',' ') in entity_types[datasetname]:
                            if datasetname in ["tacrev","tacred","retacred"]:
                                headtype = headtype.upper()
                                if headtype=="MISCELLANEOUS":
                                    headtype = "MISC"
                                else:
                                    headtype = headtype.replace(" ","_")
                                DAdata['subj_type'] = headtype
                            elif datasetname=="SciERC":
                                DAdata['subj_type'] = headtype.title()
                        else:
                            continue
                        data5 = data4[htend+len(". tail entity:"):].strip()
                    else:
                        continue
                    
                    # tail entity
                    data5lower = data5.lower()
                    if ". tail type:" in data5lower:
                        tailend = data5lower.index(". tail type:")
                        tail = data5[:tailend]
                        data6 = data5[tailend + len(". tail type:"):].strip()
                    else:
                        continue

                    # tail type
                    tailtype = data6[:-1].strip()
                    if tailtype in entity_types[datasetname] or tailtype.replace("_"," ") in entity_types[datasetname]:
                        if datasetname in ["tacrev","tacred","retacred"]:
                            headtype = headtype.upper()
                            if headtype=="MISCELLANEOUS":
                                headtype = "MISC"
                            else:
                                headtype = headtype.replace(" ","_")
                            DAdata['obj_type'] = headtype
                        elif datasetname=="SciERC":
                            DAdata['obj_type'] = headtype.title()
                    else:
                        continue
                    
                    
                    textlower = text.lower()
                    headlower = head.lower()
                    if headlower in textlower:
                        hpos1 = textlower.index(headlower)
                        hpos2 = hpos1 + len(headlower)
                        truehead = text[hpos1:hpos2]
                    else:
                        continue

                    taillower = tail.lower()
                    if taillower in textlower:
                        tpos1 = textlower.index(taillower)
                        tpos2 = tpos1 + len(taillower)
                        truetail = text[tpos1:tpos2]
                    else:
                        continue

                    DAdata['subj'] = truehead
                    DAdata['subj_start'], DAdata['subj_end'] = hpos1, hpos2
                    DAdata['obj'] = truetail
                    DAdata['obj_start'], DAdata['obj_end'] = tpos1, tpos2
                    DAdata['relation'] = k
                    f.writelines(json.dumps(DAdata, ensure_ascii=False))
                    f.write('\n')
                except:
                    pass
