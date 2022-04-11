#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs
import random
import copy


random.seed(0)

rel_set = set()


train_data = []
dev_data = []
test_data = []

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
with open('test.json') as f:
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



id2rel = {i:j for i,j in enumerate(sorted(rel_set))}
rel2id = {j:i for i,j in id2rel.items()}

with codecs.open('rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2rel, rel2id], f, indent=4, ensure_ascii=False)


with codecs.open('train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


with codecs.open('dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


with codecs.open('test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)


all_data = train_data + dev_data + test_data

print(f'train triples11:{num1}')
print(f'train Number of triples:{cnt_train}')
print(f'valid triples:{cnt_valid}')
print(f'test triples:{cnt_test}')
print('=========================')
print(f'all train_data triples Number of corpora:{len(train_data)}')
print(f'all dev_data triples:{len(dev_data)}')
print(f'all test_data triples:{len(test_data)}')
print(f'all Three together triples:{len(all_data)}')


# with codecs.open('rel2id.json', 'w', encoding='utf-8') as f:
#     json.dump([id2rel, rel2id], f, indent=4, ensure_ascii=False)


f1 = codecs.open('nyt_neg_source.txt', 'w', encoding='utf-8')

f2 = codecs.open('nyt_neg_target.txt', 'w', encoding='utf-8')



def tihuan_touwei(sents_2):
    len_sents_2 = len(sents_2)
    if len_sents_2 != 1:
        woyaoqu = random.randint(0, len_sents_2 - 1)
        sents_3 = sents_2[woyaoqu]
        woyaotihuan = random.choice([0, 2])
        tihuan_dongxi = sents_3[woyaotihuan]
        tiaochu = True
        jishujun = 0
        while tiaochu:
            jishujun += 1
            suiji1 = random.randint(0, len_sents_2 - 1)
            suiji2 = random.choice([0, 2])
            tihuan_dongxi1 = sents_2[suiji1][suiji2]
            if tihuan_dongxi1 != tihuan_dongxi:
                tiaochu = False
            if jishujun == 10:
                tiaochu = False
        xin_triple = copy.deepcopy(sents_3)
        if jishujun == 10 and tiaochu == False:
            return duoge_tihuan_rel(sents_2)
        if woyaotihuan == 0 :
            return (tihuan_dongxi1,xin_triple[1],xin_triple[2])
        if woyaotihuan == 2 :
            return (xin_triple[0],xin_triple[1],tihuan_dongxi1)




def duoge_tihuan_rel(sents_2):
    len_sents_2 = len(sents_2)
    if len_sents_2 != 1:
        woyaoqu = random.randint(0, len_sents_2 - 1)
        sents_3 = sents_2[woyaoqu]
        tihuan_dongxi = sents_3[1]
        tiaochu = True
        while tiaochu:
            quanzhong = random.random()
            if quanzhong<0.1:
                l = []
                for i in range(len_sents_2):
                    l.append(i)
                l.remove(woyaoqu)
                suiji1 = random.choice(l)
                tihuan_dongxi1 =  sents_2[suiji1][1]
            else:
                all_rel_number = len(id2rel)
                wodeid =  rel2id[tihuan_dongxi]
                l = []
                for i in range(all_rel_number):
                    l.append(i)
                l.remove(wodeid)
                suiji1 = random.choice(l)
                tihuan_dongxi1 = id2rel[suiji1]
            if tihuan_dongxi1 != tihuan_dongxi:
                tiaochu = False

        return (sents_3[0],tihuan_dongxi1,sents_3[2])



def dange_tihuan_rel(sents_2):
    len_sents_2 = len(sents_2)
    if len_sents_2 == 1:
        woyaoqu = 0
        sents_3 = sents_2[woyaoqu]
        tihuan_dongxi = sents_3[1]
        tiaochu = True
        while tiaochu:
            all_rel_number = len(id2rel)
            wodeid = rel2id[tihuan_dongxi]
            l = []
            for i in range(all_rel_number):
                l.append(i)
            l.remove(wodeid)
            suiji1 = random.choice(l)
            tihuan_dongxi1 = id2rel[suiji1]
            if tihuan_dongxi1 != tihuan_dongxi:
                tiaochu = False
        return (sents_3[0], tihuan_dongxi1, sents_3[2])



def pos_triple(sents_2):
    len_sents_2 = len(sents_2)
    woyaoqu = random.randint(0, len_sents_2 - 1)
    sents_3 = sents_2[woyaoqu]
    return (sents_3[0],sents_3[1],sents_3[2])


all_data_number = len(all_data)
for i in range(all_data_number):
    sents_1 = all_data[i]['text']
    sents_2 =  all_data[i]['triple_list']
    len_sents_2 = len(sents_2)
    pos_or_neg = random.random()
    if pos_or_neg < 0.5 :  #neg
        if len_sents_2 == 1:
            tihuan_1 = dange_tihuan_rel(sents_2)
        if len_sents_2 != 1 :
            rel_or_shiti = random.random()
            if rel_or_shiti < 0.5:
                tihuan_1 = duoge_tihuan_rel(sents_2)
            else:
                tihuan_1 = tihuan_touwei(sents_2)
        f2.write("[False]\n")
    else:   #pos
        tihuan_1 = pos_triple(sents_2)
        f2.write("[True]\n")
    xieru = sents_1+' [SEP] '+tihuan_1[0].strip()+' -> '+tihuan_1[1].strip()+' -> '+tihuan_1[2].strip()+'\n'
    f1.write(xieru)
    if (i%100 == 0):
        print("xianzai:{}".format(i))
    if (i==423):
        print("xianzai:{}".format(i))
f1.close()
f2.close()

