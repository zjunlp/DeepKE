#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs

rel_set = set()


train_data = []
dev_data = []
test_data = []
all_data = []

num1 = 0
cnt_train = 0
with open('train.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        num1 += 1
        # if not a['relationMentions']:
        #     print(a)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        cnt_train += len(line['triple_list'])
        train_data.append(line)
        for rm in a['relationMentions']:
            if rm['label'] != 'None':
                rel_set.add(rm['label'])

cnt_valid = 0
with open('valid.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        cnt_valid += len(line['triple_list'])
        dev_data.append(line)
        for rm in a['relationMentions']:
            if rm['label'] != 'None':
                rel_set.add(rm['label'])

cnt_test = 0
with open('new_test_seo.json') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        cnt_test += len(line['triple_list'])
        test_data.append(line)

all_data = train_data + dev_data + test_data

print(f'train triples11:{num1}')
print(f'train triples:{cnt_train}')
print(f'valid triples:{cnt_valid}')
print(f'test triples:{cnt_test}')
print('=========================')
print(f'all triples:{len(train_data)}')
print(f'all triples:{len(dev_data)}')
print(f'all triples:{len(test_data)}')
print(f'all triples:{len(all_data)}')



id2predicate = {i:j for i,j in enumerate(sorted(rel_set))}
predicate2id = {j:i for i,j in id2predicate.items()}


with codecs.open('rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


with codecs.open('train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


with codecs.open('dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


with codecs.open('test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)


with codecs.open('target_test_seo.txt', 'w', encoding='utf-8') as f:
    for i in tqdm(test_data):
        sent = ""
        for num in range(len(i['triple_list'])):
            if num == 0 :
                triple = i['triple_list'][num][0]+' '+'->'+' '+i['triple_list'][num][1]+' '+'->'+' '+i['triple_list'][num][2]
                sent += triple
            if num > 0 :
                triple = i['triple_list'][num][0] +' ' +'-> '+' '+ i['triple_list'][num][1] +' '+'-> '+' '+ \
                         i['triple_list'][num][2]
                sent =sent +' '+'[S2S_SEQ]'+' '+triple
        if sent == "":
            sent += 'None'
        f.write(sent+'\n')




