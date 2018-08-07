# coding=utf-8
import sys
from imp import reload
if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")
import numpy as np
import pandas as pd
import jieba
from collections import Counter
import codecs
import config
import re
from multiprocessing import Process

top_relation = config.per_com_relation
left_words = ['辞职','离职','辞去','罢免','离任','离开','退任']
for r in top_relation:
	left_words.append('前' + r)
	left_words.append('原' + r)
common_words = [u'智慧',u'其实',u'程科',u'宁波',u'于江',u'沈阳',u'海洋',u'陈述',
u'林海',u'温泉',u'营部',u'',u'江山',u'桂林',u'林路',u'蓝海',u'强力',u'高达',u'丰华',u'高峰',
u'华东',u'平衡',u'方中',u'兰克',u'方明',u'周前',u'卓越',u'路漫漫',u'任意',u'青松',u'白云',u'方舟']
company_words = [u'机器人']

all_per_com_relation = []
with codecs.open('../data/customed_dict.txt') as f:
	for line in f:
		all_per_com_relation.append(line.strip())


# 抽取关系和人的关系
def _extract_per_com_triple():
	print('生成公司和人的关系数据')
	rel_per_com = {}
	_, id2com = _get_com_map()
	pattern = re.compile("\'(.*?)\'")
	with codecs.open(config.DATA_DIR + 'kg_company_management.sql', mode='r', encoding='utf-8') as  f:
		lines = f.readlines()
	lines = lines[35:]
	for line in lines:
		s = line.strip()
		res = pattern.findall(s)
		com, per, relations = id2com[res[0]], res[1], res[5]
		for rel in relations.split(','):
			try:
				rel_per_com[rel].append([per, com])
			except:
				rel_per_com[rel] = [[per, com]]

	with codecs.open(config.DATA_DIR + 'per_com.txt', mode='r', encoding='utf-8') as  f:
		lines = f.readlines()
	lines = lines[1:]
	for line in lines:
		parts = line.strip().split('\t')
		relations, com, per = parts
		for rel in relations.split(','):
			try:
				if [per, com] not in rel_per_com[rel]:
					rel_per_com[rel].append([per, com])
			except:
				rel_per_com[rel] = [[per, com]]

	with codecs.open(config.DATA_DIR + 'rel_per_com.txt', 'w', 'utf-8') as f:
		for rel, per_coms in rel_per_com.items():
			for per_com in per_coms:
				per, com = per_com
				f.write(rel + '\t' + per + '\t' + com + '\n')

	print('生成关于公司和人的数据到stock_person.txt下面，格式[relation,person,company]')


# 公司对应简称
def _get_com_map():
	# 因为会有多个简称的出现，所以改成从abbr_com的映射,但是存在公司的简称一样的，还比较多，还是从全名到简称的映射
	com2abbr = {}  # com2abbr[公司全称] = 公司简称
	id2com = {}  # id2com[公司id] = 公司全称
	with codecs.open(config.DATA_DIR + 'com2abbr.txt', 'r', 'utf-8') as f:
		for line in f:
			parts = line.strip().split('\t')
			com2abbr[parts[0]] = parts[1]

	with codecs.open(config.DATA_DIR + 'stock.sql', mode='r', encoding='utf-8')  as  f:
		lines = f.readlines()
	lines = lines[30:]

	for line in lines:
		parts = line.strip().split("'")
		com_id, com, com_abbr = parts[1], parts[3], parts[5]
		id2com[com_id] = com
	return com2abbr, id2com


# 主要去掉\M等，不同文件的方式处理方法不同
def _clean_data(name, sep):
	f_w = codecs.open(config.DATA_DIR + 'test_' + name, 'w', encoding='utf-8')
	with codecs.open(config.DATA_DIR + name, 'r', encoding='utf-8') as f:
		count = 0
		for line in f:
			sentences = line.strip().split('\r')
			for sentence in sentences:
				info_content = sentence.strip().replace('\\n', '').split(sep)  # 注意:split的分隔符不一样，其他一样。
				if len(info_content) <= 1:
					if info_content[0].strip() != '':
						f_w.write(info_content[0].strip() + '\n')
					continue

				info = info_content[0].strip()
				content = info_content[-1].strip()  # 可能中间有多余的分隔符
				if info != '':
					f_w.write(info + '\n')

				if len(content.decode('utf-8')) > 10:
					f_w.write(content + '\n')
				count += 1
				if count % 100000 == 0:
					print(count)
		print(count)
	print('save to tmp file' + config.DATA_DIR + 'test_' + name)
	f_w.close()


def _load_rel_per_com():
	print('加载person_relation_company三元组数据')
	com2abbr, _ = _get_com_map()
	per_pool = []
	com_pool = []
	rel_per_com = {}

	with codecs.open(config.DATA_DIR + 'rel_per_com.txt', 'r', 'utf-8') as f:
		for line in f:
			try:
				rel, per, com = line.strip().split('\t')
			except:
				continue
			com_abbr = com2abbr[com]
			try:
				rel_per_com[rel].append([per, com])
			except:
				rel_per_com[rel] = [[per, com]]
			rel_per_com[rel].append([per, com_abbr])
			per_pool.append(per)
			com_pool.append(com)
			com_pool.append(com_abbr)
	# rel_per_com = [line.strip().split('\t') for line  in f]
	per_pool = list(set(per_pool))
	com_pool = list(set(com_pool))

	print('company_pool共有%d，包括公司的简称' % len(com_pool))
	print('person_pool共有%d' % len(per_pool))

	return rel_per_com, per_pool, com_pool


# 去掉句子中的！等，并用NUM代替数字
def filter_sent(dest_path, *files):
	'''
	files ： [file1_path,file2_path] 要读取的文件的列表
	dest_path : 文件保存路径
	'''
	ll = []
	for file in files:
		with codecs.open(file, 'r', 'utf-8') as f:
			for line in f:
				for s in line.strip().split('\t'):
					ll.append(s)

	fw = codecs.open(dest_path, 'w', encoding='utf-8')
	count = 0
	for line in ll:
		s = line.strip()
		if len(s) < config.MIN_CHARACTER_LEN:
			continue
		s = re.sub(r"([\!\(\)\.\?,#:@%])", r" \1 ", s)
		s = s.replace("\t|\"|“|”| |.|", "")
		s = re.sub(r"[\d|.|%]+", 'NUM', s)
		if len(s) > config.MAX_CHARACTER_LEN:
			# each_poch = 550
			# epoches = int((len(s) + each_poch -1 ) / each_poch)
			# for i in range(epoches):
			# 	sent_concat = s[i * each_poch: i * each_poch + each_poch]
			# 	fw.write(sent_concat + '\n')
			# 因为有的句子不是用句子划分的这块运行完成才会导致有的超长句子没有被整理。
			# '''
			sentences = s.split('。')
			sent_concat = ''
			for sent in sentences:
				if len(sent_concat) + len(sent) > config.MAX_CHARACTER_LEN:
					fw.write(sent_concat + '\n')
				else:
					sent_concat = ''
				sent_concat += sent
			if len(sent_concat) > config.MIN_CHARACTER_LEN:
				fw.write(sent_concat + '\n')
		# '''
		else:
			if len(s) >= config.MIN_CHARACTER_LEN:
				fw.write(s + '\n')
		count += 1
		# flag = False
		# for rel in top_relation:
		# 	if rel in s:
		# 		flag = True
		# 		break
		# if flag == False:
		# 	continue
		if count % 100000 == 0:
			print('process : %d ' % count)
	print('filter sentence end, generate file: ' + dest_path + '\n')


def sample_per_com(source_file=config.DATA_DIR + 'test.txt'):
	com2abbr, id2com = _get_com_map()
	rel_per_com, per_pool, com_pool = _load_rel_per_com()

	# 生成政府正负样本不可能是100%正确
	def run_epoch(lines, i):
		with codecs.open(config.DATA_DIR + 'per_neg_sample' + str(i) + '.txt', 'w', encoding='utf-8') as f:
			with codecs.open(config.DATA_DIR + 'per_pos_sample' + str(i) + '.txt', 'w', encoding='utf-8') as fw:
				print("start %d process" % i)
				for line_id, s in enumerate(lines):
					if line_id % 20000 == 0: print('process %d is processing %d' % (i, line_id))
					for per in per_pool:
						per_s = s.find(per)
						if per_s == -1: continue
						for com in com_pool:
							if com.find(per) != -1: continue  # 会有per是方正，com是方正科技，导致这种负样本很多
							com_s = s.find(com)
							if com_s == -1: continue
							per_e = per_s + len(per)
							com_e = com_s + len(com)
							flag = 0
							for rel, per_coms in rel_per_com.items():
								if [per, com] in per_coms:
									flag = 1
									# 这可以增加距离的判断，两实体之间距离过大，可能看不出来关系，直接标记为负样本。
									fw.write(rel + '\t' + s + '\t' + com + '\t' + per + '\t' + str(
										com_s) + '\t' + str(com_e) + '\t' + str(per_s) + '\t' + str(per_e) + '\n')
							if flag == 0:
								f.write('0\t' + s + '\t' + com + '\t' + per + '\t' + str(
									com_s) + '\t' + str(com_e) + '\t' + str(per_s) + '\t' + str(per_e) + '\n')

	ll = []
	print(source_file)
	with codecs.open(source_file, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if len(line) > 600:
				continue
			ll.append(line.replace(' ', ''))
	print('line_num = %d' % len(ll))

	# starttime = datetime.datetime.now()
	each_poch = 7000
	epoches = int((len(ll) + each_poch - 1) / each_poch)
	for i in range(epoches):
		# p = Process(target=generate_pos, args=(ll[i * each_poch: i * each_poch + each_poch], i))
		p = Process(target=run_epoch, args=(ll[i * each_poch: i * each_poch + each_poch], i))
		p.start()


def filter_neg():
	# rule1 周建灿 周建
	with open('../data/stock_person.txt') as f:
		ll = f.readlines()
	com_per_dict = {}
	for line in ll:
		parts = line.strip().split('\t')
		com, com_abbr, persons = parts[1], parts[2], parts[3:]
		com_per_dict[com] = persons
		com_per_dict[com_abbr] = persons
	with open(config.DATA_DIR + 'rel_per_com.txt') as f:
		ll = f.readlines()
	for line in ll:
		# rel,per,com = line.strip().split('\t')
		try:
			rel, per, com = line.strip().split('\t')
		except:
			continue
		try:
			if per not in com_per_dict[com]:
				com_per_dict[com].append(per)
		except:
			com_per_dict[com] = [per]

	zj_ct = 0
	source_file = '../data/per_neg.txt'
	with open(source_file) as f:
		sample = f.readlines()
	dest_file = '../data/tmp.txt'
	with open(source_file, 'w') as fw:
		with open(dest_file, 'w') as f:
			for cccc, line in enumerate(sample):
				if len(line) < 5:  continue
				parts = line.strip().split('\t')
				r, s, e1, e2, _, _, _, _ = parts
				zj_flag = 0
				try:
					for person in com_per_dict[e1]:
						if e2 in person and e2 != person and person in s:
							print(person + '\t' + e2)
							f.write(line)
							zj_ct += 1
							zj_flag = 1
							break
				except:
					fw.write(line)
				if zj_flag == 0:
					fw.write(line)

	# rule3 ： 独立董事的recall 低

	source_file = '../data/per_neg.txt'
	with open(source_file) as f:
		sample = f.readlines()
	dest_file = '../data/per_neg_drop3.txt'
	with open(source_file, 'w') as fw:
		with open(dest_file, 'w') as f:
			dlds_ct = 0
			not_dlds_ct = 0
			for cccc, line in enumerate(sample):
				if len(line) < 5:  continue
				parts = line.strip().split('\t')
				r, s, e1, e2, e11, e12, e21, e22 = parts
				e11, e12, e21, e22 = int(e11), int(e12), int(e21), int(e22)
				b, e = [e11, e22] if e11 < e21 else [e21, e12]
				tmps = s[b:e]
				s_len = len(s)
				dim = 0

				s = s.replace('非独立董事', 'fdlsd')
				if '独立董事' in tmps:
					dlds_ct += 1
					f.write(line)
				else:
					fw.write(line)
		# sample.remove(line)

		#
		# if zj_flag == 0:
		# 	fw.write(line)


def filter_pos():
	# 从708中过滤掉没有rel关键词的句子
	# source_file = '../data/708_per_pos.txt'
	source_file = '../data/per_pos.txt'
	dest_file = '../data/per_pos_drop1.txt'
	with open(source_file) as f:
		sample = f.readlines()

	with open(source_file, 'w') as fw:
		keyNotInS = 0
		with open(dest_file, 'w') as f:
			for line in sample:
				parts = line.strip().split('\t')
				r, s, e1, e2, _, _, _, _ = parts
				if r not in s:
					keyNotInS += 1
					f.write(line)
				else:
					fw.write(line)
	print('relation not in sentence: %d/%d' % (keyNotInS, len(sample)))

	# 过滤‘董事’后面是‘长’的，‘总经理’，‘总裁’前面是‘副’的，独立董事前是‘非’的
	source_file = '../data/per_pos.txt'
	dest_file = '../data/per_pos_drop2.txt'
	with open(source_file) as f:
		sample = f.readlines()

	with open(source_file, 'w') as fw:
		wrong_label_ct = 0
		right_ct = 0
		ds_ct = 0
		with open(dest_file, 'w') as f:
			for line in sample:
				parts = line.strip().split('\t')
				r, s, e1, e2, i, _, j, _ = parts
				i, j = int(i), int(j)
				b, e = [i, j + len(e2)] if i < j else [j, i + len(e1)]
				tmp_s = s[b:e]
				ds_index = tmp_s.find(r)
				if r == '董事':
					if ds_index != -1 and tmp_s[ds_index + 2] == '长':
						wrong_label_ct += 1
						ds_ct += 1
						f.write(line)
						continue
				elif r == '总经理' or r == '总裁':
					if ds_index > 0 and tmp_s[ds_index - 1] == '副':
						wrong_label_ct += 1
						f.write(line)
						continue
				elif r == '独立董事':
					if ds_index > 0 and tmp_s[ds_index - 1] == '非':
						wrong_label_ct += 1
						f.write(line)
						continue
				right_ct += 1
				fw.write(line)

	source_file = '../data/per_pos.txt'
	dest_file = '../data/per_pos_drop3.txt'
	with open(source_file) as f:
		sample = f.readlines()
	with open(source_file, 'w') as fw:
		parts = line.strip().split('\t')
		r, s, e1, e2, i, _, j, _ = parts


def seg(file_path, sample,is_training = True,filter_dynamic=True):
	jieba.load_userdict(config.EXTRA_DICT_PATH)
	with codecs.open(file_path, 'w', encoding='utf-8') as f:
		for cccc,line in enumerate(sample):
			if cccc % 10000 == 0: print('seg preprocess %d'%cccc)	
			if is_training:
				r,s,e1,e2,_,_,_,_ = line.strip().split('\t')
			else:
				r,s,e1,e2 = line.strip().split('\t')
			if is_training:
				if e1 in company_words:
					continue
				if e2 in common_words:
					continue
				# flag = 0
				# if len(e2) == 2:
				# 	for c in e2:
				# 		if c in unnormal_chars_str:
				# 			common_words.append(e2)
				# 			flag = 1
				# 			break
				# if flag == 1:
				# 	continue
			else: 
				# filter sentences for predict 
				s = re.sub(r"([\!\(\)\.\?,#:@%])", r" \1 ", s)
				s = s.replace("\t|\"|“|”| |", "")
				s = re.sub(r"[\d|.|%]+", 'NUM', s)
			s = s.replace(e1,'E1')
			s = s.replace(e2,'E2')	
			sent = jieba.cut(s)
			sent = ' '.join(sent)
			word_list = sent.split()
			sent = sent.replace('E1E2','E1 E2')
			sent = sent.replace('E2E1','E2 E1')
			sent = sent.replace('NUME1','NUM E1')
			sent = sent.replace('NUME2','NUM E2')
			flag_left = False
			if is_training == False and filter_dynamic:
				for w in left_words:
					if w in s:
						flag_left = True
						break
			if flag_left:
				continue
			if len(word_list) > config.MAX_LEN:
				print(u'%d 行分词后句子长度 > %d '%(cccc,config.MAX_LEN))
				word_list = word_list[0:config.MAX_LEN]
				sent = ' '.join(word_list)
				# continue
			if 'E1' in sent and 'E2' in sent:
				f.write('%s\t%s\t%s\t%s\t\n'%(r,sent,e1,e2))
			else:
				print('%d 行句子中没有找到实体e1/e2: %s/%s'%(cccc,e1,e2))
				print('s: %s'%(s))


def _get_closest_nword(b,e,s_seg_s,n=3):
	pass


def _get_E2_sent(e21,seg_s):
	slices = re.split('；|，|。',seg_s)
	index = 0
	for each in slices:
		word_list = each.strip().split()
		if index<=e21 and (index + len(word_list))>=e21:
			return each
		index += len(word_list)
	print('...'+seg_s+'\t'+str(e21))
	return seg_s


def _get_boundry(b,e,s_len,boundry = 8):
	dim = 0
	while True:
		if b >= 1 and dim < boundry:
			b -= 1
			dim += 1
		else:
			break
	dim = 0
	while True:
		if e < s_len and dim < boundry:
			e += 1
			dim += 1
		else:
			break
	return b,e


def _have_one_rel(seg_s):
	pattern = re.compile('法定代表人|董事长|董事|总经理|总裁|副总经理|副总裁|财务总监|监事|独立董事')
	result = re.findall(pattern,seg_s)
	if len(result) == 1:
		# print(result)
		return result[0]
	else:
		return False


def _if_intro(s):
	exp = re.compile('女博士，|博士,|硕士,|研究生,|本科学历|^E2,(女，)NUN|^E2，男|^E2先生，|^E2女士，')
	result = re.search(exp,s)
	if result:
		# print('%s\t%s' % (s[result.span()[0]:result.span()[1]], s))
		return s[result.span()[0]:result.span()[1]]
	# 看num数量，足够多就是
	else:
		return False


def _which_parallel(s):
	# 并列语句
	# 判断是不是，
	exp1 = re.compile('ORGANIZATION、|;')


def _filter_pos(pattern,lines=[],file_path=config.DATA_DIR + 'per_pos_seg_ande.txt'):
	if file_path != '':
		with open(file_path) as f:
			lines = [line.strip() for line in f]
	ct_1 = 0
	all_num = 0
	for line in lines:
		parts = line.strip().split('\t')
		r, seg_s, e1, e2, e11, e12, e21, e22 = parts
		if r != '董事':
			continue
		all_num += 1
		e11, e12, e21, e22 = int(e11), int(e12), int(e21), int(e22)
		s = seg_s.replace(' ', '')
		b, e = [e11, e22] if e11 < e21 else [e21, e12]
		tmp_s = seg_s.split()[b:e]
		e2_sent = _get_E2_sent(e21, seg_s)

		if r not in e2_sent.replace(' ',''):
			ct_1 += 1
			print(line)


def _filter_neg(pattern,lines=[],file_path=config.DATA_DIR + 'per_neg_seg_ande.txt'):
	if file_path != '':
		with open(file_path) as f:
			lines = [line.strip() for line in f]
	sample = []
	pattern_select = re.compile('选举结果|聘任|当选|发布公告|担任|出任|审议通过|换届选举|临时 股东大会')
	pattern_left = re.compile('辞职|辞去|离开|罢免|不再担任|离职')

	all_num = 0
	ct,ct_1,ct_2 = 0,0,0
	seg_s_bf = ''
	ct_1 = 0
	for line in lines:
		parts = line.strip().split('\t')
		r,seg_s,e1,e2,e11,e12,e21,e22 = parts
		e11, e12, e21, e22 = int(e11),int(e12),int(e21),int(e22)
		# if seg_s_bf == seg_s:
		# 	continue
		seg_s_bf = seg_s
		all_num += 1
		s = seg_s.replace(' ','')
		b, e = [e11, e22] if e11 < e21 else [e21, e12]
		tmp_s = seg_s.split()[b:e]
		e2_sent = _get_E2_sent(e21,seg_s)

		# 3. 离职
		# if re.search(pattern_left,s) and r in e2_sent:
		# 	ct_1 += 1
		# 	sample.append(line)
		# 	continue
		# 1. 选举结果，当选，发布公告，（担任,出任）任（relation）,审议通过，换届选举, 临时 股东大会
		if re.search(pattern_select,s) and _have_one_rel(e2_sent) == r and '。' not in tmp_s:
			ct_1 += 1
			sample.append(r+'1\t'+'\t'.join(parts[1:]))
			continue
		# 7. 介绍
		if _if_intro(s) and s.find('E1' + r) != -1:
			ct_1+=1
			sample.append(r + '1\t' + '\t'.join(parts[1:]))
			continue
		# 2. relation 和 E2近的
		if e - b > 14:
			sample.append(line)
			continue
		if re.search('E2%s|%sE2'%(r,r),s):
			if re.search('；|。|，',' '.join(tmp_s)):
				sample.append(line)
				continue
			else:
				ct_1 += 1
				# print(line + '\t' + ' '.join(tmp_s))
				sample.append(r + '1\t' + '\t'.join(parts[1:]))
				continue

		sample.append(line)
	print('%d/%d/%d' % (ct_1, ct, all_num))
	return sample


def expandEntity(file_path,sample,is_pos=False):
	with codecs.open(file_path, 'w', encoding='utf-8') as f:
		for cccc,line in enumerate(sample):
			r,seg_s,e1,e2 = line.strip().split('\t')
			e1_list = []
			e2_list = []
			word_list = seg_s.split()
			s_len = len(word_list)
			for i,w in enumerate(word_list):
				if w == 'E1':
					e1_list.append(i)
				elif w == 'E2':
					e2_list.append(i)
			if is_pos:
				have_one = False
				for i in e1_list:
					for j in e2_list:
						b,e = [i,j+len(e2)] if i<j else [j,i+len(e1)]
						tmp_segs = segs[b:e]
						if '。' not in tmp_segs:
							continue
						while True:
							if b>=1: b-=1
							else: break
						while True:
							if e+1<s_len: e+=1
							else: break
						tmp_segs = segs[b:e]
						rel_index = tmp_segs.find(rel)
						if rel_index != -1 and (abs(rel_index-e) < 10 or abs(rel_index-b) < 10 ):
							have_one = True
							f.write('%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\n'%(r,seg_s,e1,e2,i,i+1,j,j+1))
				if have_one == False:
					f.write('%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\n'%(r,seg_s,e1,e2,i,i+1,j,j+1))
			else:
				for i in e1_list:
					for j in e2_list:
						f.write('%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\n'%(r,seg_s,e1,e2,i,i+1,j,j+1))


def self_ner(file_path,sample):
	segbf = ''
	with codecs.open(file_path, 'w', encoding='utf-8') as f:
		for cccc,line in enumerate(sample):
			line = line.strip()
			parts = line.split('\t')
			r,segs,e1,e2,i,_,j,_ = parts
			# segs = unicode(segs)
			if cccc % 10000 == 0: print('self_ner %d line' % cccc)
			if segs != segbf:
				word_list = segs.split()
				ner_list = []
				for w in word_list:
					if w in all_per_com_relation:
						ner_list.append('DOMAIN')
					elif w == u'，' or w == u'。' or w == u'、' or w == u'；':
						ner_list.append(w)
					elif w == u'E1' or w == u'E2':
						ner_list.append(w)
					else:
						ner_list.append('O')
				ner_str = ' '.join(ner_list)
			f.write(line + '\t' +ner_str + '\n')


def add_self_ner(sentences):
	all_per_com_relation = []
	with codecs.open('../data/customed_dict.txt') as f:
		for line in f:
			all_per_com_relation.append(line.strip())
	ners = []
	for line in sentences:
		segs = line.strip()
		word_list = segs.split()
		ner_list = []
		for w in word_list:
			if w in all_per_com_relation:
				ner_list.append('DOMAIN')
			elif w == '，' or w == '。' or w == '、'or w == '；':
				ner_list.append(w)
			elif w == 'E1' or w == 'E2':
				ner_list.append(w)
			else:
				ner_list.append('O')
		ner_str = ' '.join(ner_list)
		ners.append(ner_str)
	print(len(ners)== len(sentences))
	return ners			


def convert2csv(file_path,sample,is_training=True):
	relations,sentences,e1s,e2s,e11s,e12s,e21s,e22s,ners = [],[],[],[],[],[],[],[],[]
	for cccc,line in enumerate(sample):
		parts = line.strip().split('\t')
		r,segs,e1,e2,e11,e12,e21,e22,ner = parts
		e11,e12,e21,e22 = int(e11), int(e12), int(e21), int(e22)
		relations.append(r) 	# 正例标记为 1
		# relations.append('0') 	# 负例标记为 0
		sentences.append(parts[1])
		e1s.append(parts[2])
		e2s.append(parts[3])
		e11s.append(parts[4])
		e12s.append(parts[5])
		e21s.append(parts[6])
		e22s.append(parts[7])
		ners.append(parts[8])

	data = pd.DataFrame(
			{'relations': relations, 'sentences': sentences, 'entity1': e1s, 'entity2': e2s, 'entity1_b': e11s,
			 'entity1_e': e12s, 'entity2_b': e21s, 'entity2_e': e22s,'ner':ners})
	columns = ['relations', 'sentences', 'entity1', 'entity2', 'entity1_b','entity1_e','entity2_b', 'entity2_e','ner']

	if is_training:
		indices = np.random.permutation(np.arange(data.shape[0]))
		data = data.iloc[indices]
	data.to_csv(file_path, index=False, sep='\t',columns = columns )


def convert2Train(pos_file,neg_file,train_path,test_path):
	pos_data = pd.read_csv(pos_file,sep='\t')
	neg_data = pd.read_csv(neg_file,sep='\t')
	train_ratio = config.train_ratio
	pos_num = pos_data.shape[0]
	neg_num = neg_data.shape[0]
	pos_bound = int(pos_num  * config.train_ratio)
	neg_bound = int(neg_num  * config.train_ratio)
	train_data = pd.DataFrame()
	test_data = pd.DataFrame()
	pos_data = pos_data.iloc[np.random.permutation(np.arange(pos_data.shape[0]))]
	neg_data = neg_data.iloc[np.random.permutation(np.arange(neg_data.shape[0]))]
	train_data = train_data.append(pos_data[0:pos_bound],ignore_index=True)
	train_data = train_data.append(neg_data[0:neg_bound],ignore_index=True)
	test_data = test_data.append(pos_data[pos_bound:],ignore_index=True)
	test_data = test_data.append(neg_data[neg_bound:],ignore_index=True)
	train_data = train_data.iloc[np.random.permutation(np.arange(train_data.shape[0]))]
	train_data.to_csv(train_path, index=False, sep='\t' )
	test_data = test_data.iloc[np.random.permutation(np.arange(test_data.shape[0]))]
	test_data.to_csv(test_path, index=False, sep='\t' )


def construct_multi_data():
	binary=config.binary
	which_relation=config.which_relation

	data1 = pd.read_csv(config.DATA_DIR + which_relation +'_train.csv',sep='\t',encoding='utf-8')
	data2 = pd.read_csv(config.DATA_DIR + which_relation + '_test.csv',sep='\t',encoding='utf-8')
	data = data1.append(data2,ignore_index=True)
	neg_data = data[data.relations=='0']

	neg_data.loc[:,'relations'] = [0] * neg_data.shape[0]
	train_data = pd.DataFrame()
	test_data = pd.DataFrame()
	if binary:
		pos_data = data[data.relations != 0]
		pos_data['relations'] = [1] * pos_data.shape[0]
		train_num = int(pos_data.shape[0] * config.train_ratio)
		train_data = train_data.append(pos_data[0:train_num],ignore_index=True)
		test_data = test_data.append(pos_data[train_num:],ignore_index=True)
	else:
		if which_relation == 'com':
			top_relation = config.com_com_relation
		else:
			top_relation = config.per_com_relation
		print('0,' + ','.join(top_relation))
		if config.CLASS_NUM != len(top_relation)+1 :
			print('类别数目不一致')
			exit()
		tmp_top_relation = top_relation.copy()
		tmp_top_relation.append('董事0')
		for relationid,rel in enumerate(tmp_top_relation):
			pos_data = data[data.relations == rel]
			pos_data.loc[:,'relations'] = [relationid+1] * pos_data.shape[0]
			train_num = int(pos_data.shape[0] * config.train_ratio)
			train_data = train_data.append(pos_data[0:train_num],ignore_index=True)
			test_data = test_data.append(pos_data[train_num:],ignore_index=True)

	train_num = int(neg_data.shape[0] * config.train_ratio)
	neg_indices = np.random.permutation(np.arange(neg_data.shape[0]))
	train_data = train_data.append(neg_data.iloc[neg_indices[0:train_num]],ignore_index=True)
	test_data = test_data.append(neg_data.iloc[neg_indices[train_num:]],ignore_index=True)

	train_data = train_data.iloc[np.random.permutation(np.arange(train_data.shape[0]))]
	test_data = test_data.iloc[np.random.permutation(np.arange(test_data.shape[0]))]

	train_data.to_csv(config.DATA_DIR + which_relation +'_multi_train.csv',sep='\t',index=False,encoding='utf-8')
	test_data.to_csv(config.DATA_DIR + which_relation +'_multi_test.csv',sep='\t',index=False,encoding='utf-8')

	print('train class number: ' + ','.join(
		str(train_data[train_data.relations == i].shape[0]) for i in range(config.CLASS_NUM)))

	print(
		'test class number: ' + ','.join(str(test_data[test_data.relations == i].shape[0]) for i in range(config.CLASS_NUM)))


def build_dict(sentences):
	# 统计每一个单词出现的次数
	word_count = Counter()
	for s in sentences:
		if s == '':
			continue
		for w in s.split():
			if w == '' or w == None:
				continue
			word_count[w] += 1
	vocabs = word_count.most_common()
	word_dict = {w[0]: i + 1 for (i, w) in enumerate(vocabs)}

	with open(config.DATA_DIR + 'word_dict.txt', 'w') as ofile:
		for w in word_dict:
			ofile.write(str(word_dict[w]) + '\t' + w + '\t' + str(word_count[w]) + '\n')
	return word_dict


def get_initial_sample():
	_clean_data('news.txt', '　　')
	_clean_data('announcement.txt', '    ')
	files = [config.DATA_DIR + 'test_announcement.txt', config.DATA_DIR + 'test_news.txt']
	filter_sent(config.DATA_DIR + 'test.txt', *files)
	_extract_per_com_triple()
	# 采样：耗时较长
	sample_per_com(config.DATA_DIR + 'test.txt')
	# 上述采样结束后
	# 命令行执行  cat per_neg_sample* > per_neg.txt
	# 命令行执行	cat per_pos_sample* > per_pos.txt
	# 之后进行第一轮过滤
	filter_neg()
	filter_pos()


def train_preprocess():
	# step1 seg
	pos_file = config.DATA_DIR + 'per_pos.txt'		
	neg_file = config.DATA_DIR + 'per_neg.txt'
	with codecs.open(neg_file, 'r', encoding='utf-8') as f:
		neg_sample = f.readlines()
	with codecs.open(pos_file, 'r', encoding='utf-8') as f:
		pos_sample = f.readlines()
	pos_sample = list(set(pos_sample))
	neg_sample = list(set(neg_sample))
	print('pos_num:%d'%len(pos_sample))
	print('neg_num:%d'%len(neg_sample))
	seg(file_path='../data/per_neg_seg.txt', sample=neg_sample)
	seg(file_path='../data/per_pos_seg.txt', sample=pos_sample)

	# step2 expandEntity
	pos_file = config.DATA_DIR + 'per_pos_seg.txt'
	neg_file = config.DATA_DIR + 'per_neg_seg.txt'
	with codecs.open(neg_file, 'r', encoding='utf-8') as f:
		neg_sample_seg = f.readlines()
	with codecs.open(pos_file, 'r', encoding='utf-8') as f:
		pos_sample_seg = f.readlines()
	print('pos_num:%d'%len(pos_sample_seg))
	print('neg_num:%d' % len(neg_sample_seg))
	sentences1 = [line.strip().split('\t')[1] for line in neg_sample_seg]
	sentences2 = [line.strip().split('\t')[1] for line in pos_sample_seg]
	sentences1.extend(sentences2)
	for rel in config.per_com_relation:
		data = pd.read_csv('../result/'+rel+'.csv',sep='\t')
		sentencesi = data.sentences
		sentences1.extend(sentencesi)
	word_dict = build_dict(sentences1)
	print('单词数: %d'%(len(word_dict)))

	expandEntity('../data/per_neg_seg_ande.txt',neg_sample_seg)
	expandEntity('../data/per_pos_seg_ande.txt',pos_sample_seg,is_pos=True)

	# train : step3 ner 
	# cat per_pos_seg_rule1.txt per_neg_seg_rule1.txt > per_pos.txt
	pos_file = config.DATA_DIR + 'per_pos_seg_ande.txt'
	neg_file = config.DATA_DIR + 'per_neg_seg_ande.txt'
	with codecs.open(neg_file, 'r', encoding='utf-8') as f:
		neg_sample = f.readlines()
	with codecs.open(pos_file, 'r', encoding='utf-8') as f:
		pos_sample = f.readlines()

	self_ner('../data/per_pos_seg_ner.txt',pos_sample)
	self_ner('../data/per_neg_seg_ner.txt',neg_sample)

	# train: step4 filter，后续会在expandEntity加入
	# without filtering
	pos_file = config.DATA_DIR + 'per_pos_seg_ner.txt'
	neg_file = config.DATA_DIR + 'per_neg_seg_ner.txt'

	# step5: convert pos.txt and neg.txt to pos.csv and neg.csv
	with codecs.open(neg_file, 'r', encoding='utf-8') as f:
		neg_sample = f.readlines()
	with codecs.open(pos_file, 'r', encoding='utf-8') as f:
		pos_sample = f.readlines()

	convert2csv('../data/per_pos.csv',pos_sample,is_training=True)
	convert2csv('../data/per_neg.csv', neg_sample, is_training=True)

	# step6: generate train.csv and test.csv
	pos_file = config.DATA_DIR + 'per_pos.csv'
	neg_file = config.DATA_DIR + 'per_neg.csv'
	train_path = config.DATA_DIR + 'per_train.csv'
	test_path = config.DATA_DIR + 'per_test.csv'
	convert2Train(pos_file,neg_file,train_path,test_path)
	# step6
	# construct_multi_data(pos_file,neg_file)
	construct_multi_data()


# 输入预测样本格式：relation sentence e1 e2

def predict_preprocess():
	# predict: step1 segment
	file_path = '../data/predict.txt'
	with codecs.open(file_path, 'r', encoding='utf-8') as f:
		sample = f.readlines()

	# check predict.txt file format
	print('check predict.txt file format')
	for cccc,line in enumerate(sample):
		try:
			r,s,e1,e2, = line.strip().split('\t')
		except:
			print('%d: %s'%(cccc,line))

	seg('../data/predict_seg.txt',sample, is_training = False,filter_dynamic=True)

	# predict: step2 seg_ande
	file_path = '../data/predict_seg.txt'
	with codecs.open(file_path, 'r', encoding='utf-8') as f:
		sample = f.readlines()
	expandEntity('../data/predict_seg_ande.txt',sample)

	# predict: step3 ner
	file_path = '../data/predict_seg_ande.txt'
	with codecs.open(file_path, 'r', encoding='utf-8') as f:
		sample = f.readlines()

	self_ner('../data/predict_seg_ner.txt',sample)

	# predict: step4 convert2csv
	pos_file = config.DATA_DIR + 'predict_seg_ner.txt'
	with codecs.open(pos_file, 'r', encoding='utf-8') as f:
		pos_sample = f.readlines()
	convert2csv('../data/predict.csv',pos_sample,is_training=False)


if __name__ == '__main__':
	get_initial_sample()
	train_preprocess()
	# 可执行 python train.py
	predict_preprocess()
	# 可执行 python predict.py