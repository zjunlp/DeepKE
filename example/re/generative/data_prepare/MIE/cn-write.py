import argparse
import os
from src import Dictionary, Ontology, Data, MIE, evaluate

dictionary = Dictionary()
dictionary.load('./data/dictionary.txt')


ontology = Ontology(dictionary)
ontology.add_raw('./data/ontology.json', '状态')
ontology.add_examples('./data/example_dict.json')

data = Data(100, dictionary, ontology)
data.add_raw('train', './data/train.json', 'window')
data.add_raw('test', './data/test.json', 'window')
data.add_raw('dev', './data/dev.json', 'window')


#The following code generates source sentences for training set and test set respectively
# fo = open("./data/train_source_cn.txt", "w",encoding='utf8')
# for source_sent in data.datasets['train']['source']:
#     fo.write(source_sent+'\n')
# for source_sent in data.datasets['dev']['source']:
#     fo.write(source_sent + '\n')
# fo.close()
#
# fo = open("./data/test_source_cn.txt", "w",encoding='utf8')
# for source_sent in data.datasets['test']['source']:
#     fo.write(source_sent+'\n')
# fo.close()

#The following code generates target sentences for training set and test set
# fo = open("./data/train_target_cn.txt", "w",encoding='utf8')
# for source_sent in data.datasets['train']['target']:
#     fo.write(source_sent+'\n')
# for source_sent in data.datasets['dev']['target']:
#     fo.write(source_sent + '\n')
# fo.close()
#
# fo = open("./data/test_target_cn.txt", "w",encoding='utf8')
# for source_sent in data.datasets['test']['target']:
#     fo.write(source_sent+'\n')
# fo.close()

#The following code is in the unilm-cn format
fo = open("./data/train_cn.json", "w",encoding='utf8')
num = len(data.datasets['train']['source'])
for i in range(num):
    sents = {'src_text':data.datasets['train']['source'][i],'tgt_text':data.datasets['train']['target'][i]}
    fo.write(str(sents)+'\n')
num = len(data.datasets['dev']['source'])
for i in range(num):
    sents = {'src_text':data.datasets['dev']['source'][i],'tgt_text':data.datasets['dev']['target'][i]}
    fo.write(str(sents)+'\n')
fo.close()













