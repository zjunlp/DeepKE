import re

RE_triple0 = re.compile('(\([^\(\)]*\)?)')
RE_triple3 = re.compile('(\{[^\{\}]*\}?)')
RE_triple3_head = re.compile("'head':[\s]?'([^\'\']*?)',")
RE_triple3_rel = re.compile("'relation':[\s]?'([^\'\']*?)',")
RE_triple3_tail = re.compile("'tail':[\s]?'([^\'\']*?)'}")

NAN = "NAN" 

def nan(s):
    if s.strip() == "":
        return NAN
    return s

prefix = "输入中包含的关系三元组是：\n"


def relation_convert_target0(rels):
    if len(rels) == 0:
        return prefix + NAN 
    output_text = []
    for rel in rels:
        output_text.append('(' + ','.join([nan(rel['head']), nan(rel['relation']), nan(rel['tail'])]) + ')')
    output_text = ','.join(output_text)
    return prefix + output_text 


def relation_convert_target0_en(rels):
    if len(rels) == 0:
        return NAN 
    output_text = []
    for rel in rels:
        output_text.append(f"({nan(rel['head'])}, {nan(rel['relation'])}, {nan(rel['tail'])})")
    output_text = ','.join(output_text)
    return output_text 
    
    
def relation_convert_target1(rels):
    if len(rels) == 0:
        return prefix + NAN 
    output_text = []
    for rel in rels:
        output_text.append(f"头实体是{nan(rel['head'])}\n关系是{nan(rel['relation'])}\n尾实体是{nan(rel['tail'])}\n\n")
    output_text = ''.join(output_text)
    return prefix + output_text


def relation_convert_target1_en(rels):
    if len(rels) == 0:
        return NAN 
    new_rels = []
    for rel in rels:
        new_rels.append({"head":nan(rel['head']), "relation":nan(rel['relation']), "tail":nan(rel['tail'])})
    return str(new_rels) 


def relation_convert_target2(rels):
    if len(rels) == 0:
        return prefix + NAN 
    output_text = []
    for rel in rels:
        output_text.append(f"{nan(rel['relation'])}：{nan(rel['head'])},{nan(rel['tail'])}\n")
    output_text = ''.join(output_text)
    return prefix + output_text 


def relation_convert_target3(rels):
    if len(rels) == 0:
        return prefix + NAN 
    new_rels = []
    for rel in rels:
        new_rels.append({"head":nan(rel['head']), "relation":nan(rel['relation']), "tail":nan(rel['tail'])})
    return prefix + str(new_rels) 


relation_template_zh =  {
    0:'已知候选的关系列表：{s_schema}，请你根据关系列表，从以下输入中抽取出可能存在的头实体与尾实体，并给出对应的关系三元组。请按照{s_format}的格式回答。',
    1:'我将给你个输入，请根据关系列表：{s_schema}，从输入中抽取出可能包含的关系三元组，并以{s_format}的形式回答。',
    2:'我希望你根据关系列表从给定的输入中抽取可能的关系三元组，并以{s_format}的格式回答，关系列表={s_schema}。',
    3:'给定的关系列表是{s_schema}\n根据关系列表抽取关系三元组，在这个句子中可能包含哪些关系三元组？请以{s_format}的格式回答。',
} 

relation_template_en =  {
    0:'Identify the head entities (subjects) and tail entities (objects) in the following text and provide the corresponding relation triples from relation list {s_schema}. Please provide your answer as a list of relation triples in the form of {s_format}.',
    1:'From the given text, extract the possible head entities (subjects) and tail entities (objects) and give the corresponding relation triples. The relations are {s_schema}. Please format your answer as a list of relation triples in the form of {s_format}.', 
}

relation_int_out_format_zh = {
    0:['"(头实体,关系,尾实体)"', relation_convert_target0],
    1:['"头实体是\n关系是\n尾实体是\n\n"', relation_convert_target1],
    2:['"关系：头实体,尾实体\n"', relation_convert_target2],
    3:["JSON字符串[{'head':'', 'relation':'', 'tail':''}, ]", relation_convert_target3],
}

relation_int_out_format_en = {
    0:['(Subject, Relation, Object)', relation_convert_target0_en],
    1:["{'head':'', 'relation':'', 'tail':''}", relation_convert_target1_en],
}






def clean(s):
    return s.strip()


def replace_nomiss(s):
    return s.replace(prefix, "")


def rte_post_process0_en(result):    
    def clean(spt, i):
        try:
            s = spt[i].strip()
        except IndexError:
            s = ""
        if i == 0 and len(s) != 0 and s[0] == '(':
            s = s[1:]
        if i == 2 and len(s) != 0 and s[-1] == ')':
            s = s[:-1]
        return s
    
    matches = re.findall(RE_triple0, result)
    new_record = []
    for m in matches:
        spt = m.split(",")
        if len(spt) != 3:
            continue
        head, rel, tail = clean(spt, 0), clean(spt, 1), clean(spt, 2)
        if head == NAN or rel ==NAN or tail == NAN:
            continue
        new_record.append([head, rel, tail])

    return new_record


def rte_post_process1_en(result):        
    def get_span(text, template):
        match = re.findall(template, text)
        try: 
            match = match[0]
        except IndexError:
            match = ""
        return match.strip()

    matches = re.findall(RE_triple3, result)
    new_record = []
    for m in matches:
        head = get_span(m, RE_triple3_head)
        rel = get_span(m, RE_triple3_rel)
        tail = get_span(m, RE_triple3_tail)
        print(head, rel, tail)
        if head == NAN or rel ==NAN or tail == NAN:
            continue
        new_record.append([head, rel, tail])

    return new_record


def rte_post_process0_zh(result):    
    def clean(spt, i):
        try:
            s = spt[i].strip()
        except IndexError:
            s = ""
        if i == 0 and len(s) != 0 and s[0] == '(':
            s = s[1:]
        if i == 2 and len(s) != 0 and s[-1] == ')':
            s = s[:-1]
        return s
    
    rst = replace_nomiss(result)
    matches = re.findall(RE_triple0, rst)
    new_record = []
    for m in matches:
        spt = m.split(",")
        if len(spt) != 3:
            continue
        head, rel, tail = clean(spt, 0), clean(spt, 1), clean(spt, 2)
        if head == "" or rel == "" or tail == "":
            continue
        if head == NAN or tail == NAN:
            continue
        new_record.append([head, rel, tail])

    return new_record


def rte_post_process1_zh(result):        
    def clean(spt, i):
        try:
            s = spt[i].strip()
        except IndexError:
            s = ""
        if i == 0 and len(s) != 0:
            s = s.replace("头实体是", "")
        if i == 1 and len(s) != 0:
            s = s.replace("关系是", "")
        if i == 2 and len(s) != 0:
            s = s.replace("尾实体是", "")
        return s

    rst = replace_nomiss(result)
    matches = rst.split("\n\n")
    new_record = []
    for m in matches:
        spt = m.split("\n")
        if len(spt) != 3:
            continue
        head = clean(spt, 0)
        rel = clean(spt, 1)
        tail = clean(spt, 2)
        if head == "" or rel == "" or tail == "":
            continue
        if head == NAN or tail == NAN:
            continue
        new_record.append([head, rel, tail])

    return new_record


def rte_post_process2_zh(result):        
    def clean(spt):
        spt = spt.strip()
        return spt  
    
    rst = replace_nomiss(result)
    matches = rst.split("\n")
    new_record = []
    for m in matches:
        index1 = m.find("：")
        index2 = m.find(",")
        rel, head, tail = clean(m[:index1]), clean(m[index1+1:index2]), clean(m[index2+1:])
        if head == "" or rel == "" or tail == "":
            continue
        if head == NAN or tail == NAN:
            continue
        new_record.append([head, rel, tail])

    return new_record



def rte_post_process3_zh(result):        
    def get_span(text, template):
        match = re.findall(template, text)
        try: 
            match = match[0]
        except IndexError:
            match = ""
        return match.strip()

    rst = replace_nomiss(result)
    matches = re.findall(RE_triple3, rst)
    new_record = []
    for m in matches:
        head = get_span(m, RE_triple3_head)
        rel = get_span(m, RE_triple3_rel)
        tail = get_span(m, RE_triple3_tail)
        if head == "" or rel == "" or tail == "":
            continue
        if head == NAN or tail == NAN:
            continue
        new_record.append([head, rel, tail])

    return new_record



def re_post_process_zh(output):
    clean_output = clean(output)
    if clean_output == NAN:
        return []
    if 'head' in output and 'relation' in output and 'tail' in output and "'" in output and '{' in output and '}' in output and ':' in output:
        kg = rte_post_process3_zh(output)
    elif '头实体是' in output and '尾实体是' in output and '关系是' in output:
        kg = rte_post_process1_zh(output)
    elif '(' in output and ')' in output and ',' in output:
        kg = rte_post_process0_zh(output)
    elif "：" in output and "," in output and "\n" in output:
        kg = rte_post_process2_zh(output)
    else:
        kg = None
    return kg


def re_post_process_en(output):
    clean_output = clean(output)
    if clean_output == NAN:
        return []
    if "head" in output and "relation" in output and "tail" in output and "'" in output and '{' in output and '}' in output and ':' in output:
        kg = rte_post_process1_en(output)
    elif '(' in output and ')' in output and ',' in output:
        kg = rte_post_process0_en(output)
    else:
        kg = None
    return kg

