# coding=utf-8
#

def preprocess(auto_rec_entity=False):
	'''
	整体是将predict.txt转为能输入模型的predict.csv
	predict.txt目前是[sent,e1,e2]的格式；另外是[sent]考虑的是真是场景做kgc一般只是给sent

	1. predict.txt首先需要clean，（注意如果是[sent,e1,e2]会出现，pos1和pos2的比较靠句子后面而句子长度过长）
	2. 可能需要ner
	3. relations,get_pos1,get_pos2, replace_entity, ner, mask
	基本上很多代码在tools中有，
	注意加载数据集时候是和predict数据集也会relation，注意在predict的预处理上如果没有relation，就加载个0
	:param auto_rec_entity:是否需要程序自己通过ner识别出来两个实体（注意如果用ner会识别出来多个）
	:return:
	'''
