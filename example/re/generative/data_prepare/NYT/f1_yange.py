#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs


open_shuchu_name = 'test_data_decorder.txt'
source_file_name = 'test_date_target.txt'

real_test_data = []

cnt_test = 0
num1 = 0
with open('test.json') as f:
    for l in tqdm(f):
        num1 += 1
        a = json.loads(l)
        if not a['relationMentions']:
            continue
        line = {
                'text': a['sentText'].lstrip('\"').strip('\r\n').rstrip('\"'),
                'triple_list': [(i['em1Text'], i['label'].split('/')[-1], i['em2Text']) for i in a['relationMentions'] if i['label'] != 'None']
               }
        if not line['triple_list']:
            continue
        cnt_test += len(line['triple_list'])
        real_test_data.append(line)

print(f"all test triple is :{cnt_test}")
print(f"all test tiaoshu is :{num1}")

num2 = 0
with open(open_shuchu_name) as f:
    for l in tqdm(f):
        # test_list = l.split('[S2S_SEQ]')
        # print(test_list)
        num2 += 1


num3 = 0
pipei = 0
kk = open('error_write.txt', 'w')


source_sentence = "NUll"
source_file = open(source_file_name,'r')

with open(open_shuchu_name) as f:
    for l in tqdm(f):
        source_sentence = source_file.readline()
        test_text = str(l)
        test_text = test_text.replace(' ','')
        test_ceshi_triple = test_text.split("[S2S_SEQ]")
        # print(test_text)
        # print(test_ceshi_triple)
        real_triple = real_test_data[num3]['triple_list']
        test_ceshi_biaozhu = [0] * len(test_ceshi_triple)
        for i in range(len(real_triple)):
            relation = real_triple[i][1].replace(' ','')
            toushiti = real_triple[i][0]
            weishiti = real_triple[i][2]
            # zhengque = relation in test_text
            zhengque = False
            for j in range(len(test_ceshi_triple)):
                if relation in test_ceshi_triple[j]:
                    if toushiti in test_ceshi_triple[j]:
                        if weishiti in test_ceshi_triple[j]:
                            # print(test_ceshi_triple[j])
                            test_ceshi_biaozhu [j] = 1
                            zhengque = True
            if zhengque:
                pipei = pipei + 1

        for m in range(len(test_ceshi_triple)):
            if test_ceshi_biaozhu [m] ==0 :
                kk.write(f"The source sentence is：{source_sentence.strip()}\n")
                kk.write(f"The output statement is：{l.strip()}\n")
                kk.write(f"The error statement is：{test_ceshi_triple[m].strip()}\n")
                kk.write(f"The correct sentence is：{real_triple}\n")
                kk.write(f"-----------------------------------\n")
        num3 += 1


zhunque = 0
num4 = 0
num3 = 0
with open(open_shuchu_name) as f:
    for l in tqdm(f):
        test_text = str(l)
        test_text = test_text.replace(' ','')
        test_ceshi_triple = test_text.split("[S2S_SEQ]")
        real_triple = real_test_data[num3]['triple_list']
        num4 += len(test_ceshi_triple)
        for i in range(len(test_ceshi_triple)):
            if test_ceshi_triple[i] == '':
                num4 -= 1
                continue
            if '->' not in test_ceshi_triple[i]:
                num4 -= 1
                continue
        for i in range(len(real_triple)):
            relation = real_triple[i][1]
            toushiti = real_triple[i][0]
            weishiti = real_triple[i][2]
            zhengque = relation in test_text
        num3 += 1


print(f"all all_model90 tiaoshu is :{num2}")
print(f"total zhaohui  {pipei} can pipei")
print(f"total zhunque {num4} can zhunque")
print("==================")
print("prec: %.4f "%(pipei/num4))
print("zhaohui: %.4f "%(pipei/cnt_test))
f1 = 2*(pipei/num4)*(pipei/cnt_test)/((pipei/num4)+(pipei/cnt_test))
print("f1 : %.4f"%(f1))