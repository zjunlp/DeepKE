import re

NER_triple0 = re.compile('(\([^\(\)]*\)?)')
NER_triple3 = re.compile('(\{[^\{\}]*\}?)')
NER_triple3_entity_en = re.compile("'Entity':[\s]?'([^\'\']*?)',")
NER_triple3_type_en = re.compile("'Type':[\s]?'([^\'\']*?)',")
NER_triple3_entity_zh = re.compile("'entity':[\s]?'([^\'\']*?)',")
NER_triple3_type_zh = re.compile("'entity_type':[\s]?'([^\'\']*?)',")


NAN = "NAN" 

def nan(s):
    if s.strip() == "":
        return NAN
    return s

prefix = "输入中包含的实体是：\n"

def entity_convert_target0(entities):
    if len(entities) == 0:
        return prefix + NAN 
    output_text = []
    for entity in entities:
        output_text.append('(' + ','.join([nan(entity['entity']), nan(entity['entity_type'])]) + ')')
    output_text = ','.join(output_text)
    return prefix + output_text



def entity_convert_target0_en(entities):
    if len(entities) == 0:
        return NAN 
    output_text = []
    for entity in entities:
        output_text.append(f"({nan(entity['entity'])}, {nan(entity['entity_type'])})")
    output_text = ','.join(output_text)
    return output_text


def entity_convert_target1(entities):
    if len(entities) == 0:
        return prefix + NAN 
    output_text = []
    for entity in entities:
        output_text.append(f"实体是{nan(entity['entity'])}\n实体类型是{nan(entity['entity_type'])}\n\n")
    output_text = ''.join(output_text)
    return prefix + output_text 


def entity_convert_target1_en(entities):
    if len(entities) == 0:
        return NAN 
    new_entities = []
    for ent in entities:
        new_entities.append({"Entity":nan(ent['entity']), "Type":nan(ent['entity_type'])})
    return str(new_entities) 


def entity_convert_target2(entities):
    if len(entities) == 0:
        return prefix + NAN 
    output_text = []
    for entity in entities:
        output_text.append(f"{nan(entity['entity_type'])}：{nan(entity['entity'])}\n")
    output_text = ''.join(output_text)
    return prefix + output_text 


def entity_convert_target3(entities):
    if len(entities) == 0:
        return prefix + NAN 
    new_entities = []
    for ent in entities:
        new_entities.append({"entity":nan(ent['entity']), "entity_type":nan(ent['entity_type'])})
    return prefix + str(new_entities) 


entity_template_zh =  {
    0:'已知候选的实体类型列表：{s_schema}，请你根据实体类型列表，从以下输入中抽取出可能存在的实体。请按照{s_format}的格式回答。',
    1:'我将给你个输入，请根据实体类型列表：{s_schema}，从输入中抽取出可能包含的实体，并以{s_format}的形式回答。',
    2:'我希望你根据实体类型列表从给定的输入中抽取可能的实体，并以{s_format}的格式回答，实体类型列表={s_schema}。',
    3:'给定的实体类型列表是{s_schema}\n根据实体类型列表抽取，在这个句子中可能包含哪些实体？你可以先别出实体, 再判断实体类型。请以{s_format}的格式回答。',
}

entity_template_en =  {
    0:'Identify the entities and types in the following text and where entity type list {s_schema}. Please provide your answerin the form of {s_format}.',
    1:'From the given text, extract the possible entities and types . The types are {s_schema}. Please format your answerin the form of {s_format}.', 
}

entity_int_out_format_zh = {
    0:['"(实体,实体类型)"', entity_convert_target0],
    1:['"实体是\n实体类型是\n\n"', entity_convert_target1],
    2:['"实体：实体类型\n"', entity_convert_target2],
    3:["JSON字符串[{'entity':'', 'entity_type':''}, ]", entity_convert_target3],
}

entity_int_out_format_en = {
    0:['(Entity, Type)', entity_convert_target0_en],
    1:["{'Entity':'', 'Type':''}", entity_convert_target1_en],
}




def clean(s):
    return s.strip()

def replace_nomiss(s):
    return s.replace(prefix, "")



def ner_post_process0_en(result):    
    def clean(spt, i):
        try:
            s = spt[i].strip()
        except IndexError:
            s = ""
        if i == 0 and len(s) != 0 and s[0] == '(':
            s = s[1:]
        if i == 1 and len(s) != 0 and s[-1] == ')':
            s = s[:-1]
        return s
    
    matches = re.findall(NER_triple0, result)
    new_record = []
    for m in matches:
        spt = m.split(",")
        if len(spt) != 2:
            continue
        entity, type = clean(spt, 0), clean(spt, 1)
        if entity == NAN or type == NAN:
            continue
        new_record.append([entity, type])
    return new_record


def ner_post_process1_en(result):        
    def get_span(text, template):
        match = re.findall(template, text)
        try: 
            match = match[0]
        except IndexError:
            match = ""
        return match.strip()

    matches = re.findall(NER_triple3, result)
    new_record = []
    for m in matches:
        entity = get_span(m, NER_triple3_entity_en)
        type = get_span(m, NER_triple3_type_en)
        if entity == NAN or type == NAN:
            continue
        new_record.append([entity, type])
    return new_record



def ner_post_process0_zh(result):    
    def clean(spt, i):
        try:
            s = spt[i].strip()
        except IndexError:
            s = ""
        if i == 0 and len(s) != 0 and s[0] == '(':
            s = s[1:]
        if i == 1 and len(s) != 0 and s[-1] == ')':
            s = s[:-1]
        return s
    
    matches = re.findall(NER_triple0, result)
    new_record = []
    for m in matches:
        spt = m.split(",")
        if len(spt) != 2:
            continue
        entity, type = clean(spt, 0), clean(spt, 1)
        if entity == NAN or type ==NAN:
            continue
        new_record.append([entity, type])

    return new_record



def ner_post_process1_zh(result):        
    def clean(spt, i):
        try:
            s = spt[i].strip()
        except IndexError:
            s = ""
        if i == 0 and len(s) != 0:
            s = s.replace("实体是", "")
        if i == 1 and len(s) != 0:
            s = s.replace("实体类型是", "")
        return s

    rst = replace_nomiss(result)
    matches = rst.split("\n\n")
    new_record = []
    for m in matches:
        spt = m.split("\n")
        if len(spt) != 2:
            continue
        entity = clean(spt, 0)
        type = clean(spt, 1)
        if entity == NAN or type ==NAN:
            continue
        new_record.append([entity, type])

    return new_record


def ner_post_process2_zh(result):        
    def clean(spt):
        spt = spt.strip()
        return spt  
    
    rst = replace_nomiss(result)
    matches = rst.split("\n")
    new_record = []
    for m in matches:
        spt = m.split("：")
        if len(spt) != 2:
            continue
        type, entity = clean(spt[0]), clean(spt[1])
        if entity == NAN or type ==NAN:
            continue
        new_record.append([entity, type])

    return new_record


def ner_post_process3_zh(result):        
    def get_span(text, template):
        match = re.findall(template, text)
        try: 
            match = match[0]
        except IndexError:
            match = ""
        return match.strip()

    matches = re.findall(NER_triple3, result)
    new_record = []
    for m in matches:
        entity = get_span(m, NER_triple3_entity_zh)
        type = get_span(m, NER_triple3_type_zh)
        if entity == NAN or type ==NAN:
            continue
        new_record.append([entity, type])

    return new_record



def ner_post_process_zh(output):
    clean_output = clean(output)
    if clean_output == NAN:
        return []
    if 'entity' in output and 'entity_type' in output and 'tail' in output and "'" in output and '{' in output and '}' in output and ':' in output:
        kg = ner_post_process3_zh(output)
    elif '实体是' in output and '实体类型是' in output:
        kg = ner_post_process1_zh(output)
    elif '(' in output and ')' in output and ',' in output:
        kg = ner_post_process0_zh(output)
    elif "：" in output and "\n" in output:
        kg = ner_post_process2_zh(output)
    else:
        kg = None
    return kg


def ner_post_process_en(output):
    clean_output = clean(output)
    if clean_output == NAN:
        return []
    if "Entity" in output and "Type" in output and "'" in output and '{' in output and '}' in output and ':' in output:
        kg = ner_post_process1_en(output)
    elif '(' in output and ')' in output and ',' in output:
        kg = ner_post_process0_en(output)
    else:
        kg = None
    return kg