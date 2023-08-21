import re
NAN = "NAN" 
EE_triple0_en = re.compile('(\([^\(\)]*\)?)')
EE_triple0_zh = re.compile('(\([^\(\)]*\]\)?)')
EE_triple0_arg_zh = re.compile('(\([^\(\)]*\)?)')

def nan(s):
    if s.strip() == "":
        return NAN
    return s

prefix = "输入中包含的事件是：\n"

def event_convert_target0(events):
    if len(events) == 0:
        return  prefix + NAN
    output_text = []
    for event in events:
        args = []
        for arg in event['arguments']:
            args.append('(' + ','.join([nan(arg['argument']), nan(arg['role'])]) + ')')
        if len(args) == 0:
            args.append('(' + ','.join([NAN, NAN]) + ')')
        args = '[' + ','.join(args) + ']'
        output_text.append('(' + ','.join([nan(event['event_trigger']), nan(event['event_type']), args]) + ')')
    output_text = ','.join(output_text) 
    return prefix + output_text 


def event_convert_target0_en(events):
    if len(events) == 0:
        return NAN
    output_text = []
    for event in events:
        args = []
        for arg in event['arguments']:
            args.append(f"{nan(arg['role'])}#{nan(arg['argument'])}")
        if len(args) == 0:
            args.append('NAN#NAN')
        args = '; '.join(args)
        output_text.append(f"({nan(event['event_type'])}, {nan(event['event_trigger'])}, {args})")
    output_text = ','.join(output_text) 
    return output_text 
    

def event_convert_target1(events):
    if len(events) == 0:
        return  prefix + NAN 
    output_text = []
    for event in events:
        args = []
        for arg in event['arguments']:
            args.append(f"\t事件论元是{nan(arg['argument'])},论元角色是{nan(arg['role'])}")
        if len(args) == 0:
            args.append(f"\t事件论元是{NAN},论元角色是{NAN}")
        args = '\n'.join(args)
        output_text.append( f"事件触发词是{nan(event['event_trigger'])},事件类型是{nan(event['event_type'])}" + "\n" + args)
    output_text = '\n'.join(output_text)
    return prefix + output_text + "\n"


def event_convert_target2(events):
    if len(events) == 0:
        return  prefix + NAN 
    output_text = []
    for event in events:
        args = []
        for arg in event['arguments']:
            args.append(f"\t{nan(arg['role'])}：{nan(arg['argument'])}")
        if len(args) == 0:
            args.append(f"\t{NAN}：{NAN}")
        args = '\n'.join(args)
        output_text.append( f"{nan(event['event_type'])}：{nan(event['event_trigger'])}" + "\n" + args)
    output_text = '\n'.join(output_text)
    return prefix + output_text + "\n"


def event_convert_target3(events):
    if len(events) == 0:
        return  prefix + NAN 
    new_events = []
    for event in events:
        new_args = []
        for arg in event['arguments']:
            new_args.append({'argument': nan(arg['argument']), 'role': nan(arg['role'])})
        new_events.append({"event_trigger":nan(event['event_trigger']), "event_type":nan(event['event_type']), "arguments": new_args})
    return prefix + str(new_events) 


event_template_zh =  {
    0:'已知候选的事件类型列表：{s_schema1}，论元角色列表：{s_schema2}，请你根据事件类型列表和论元角色列表，从以下输入中抽取出可能存在的事件。请按照{s_format}的格式回答。',
    1:'我将给你个输入，请根据事件类型列表：{s_schema1}，论元角色列表：{s_schema2}，从输入中抽取出可能包含的事件，并以{s_format}的形式回答。',
    2:'我希望你根据事件类型列表和论元角色列表从给定的输入中抽取可能的事件，并以{s_format}的格式回答，事件类型列表={s_schema1}，论元角色列表={s_schema2}。',
    3:'给定的事件类型列表是{s_schema1}，论元角色列表是{s_schema2}\n根据事件类型列表和论元角色列表抽取，在这个句子中可能包含哪些事件？请以{s_format}的格式回答。',
}

event_template_en = {
    0:'Identify the event information in the following text and provide the corresponding triples from event type list {s_schema1} and role list {s_schema2}. Please provide your answer as a list of triples in the form of {s_format}.',
    1:'From the given text, extract the possible event information and give the corresponding triples. The event types are {s_schema1} and the role list are {s_schema2}. Please format your answer as a list of triples in the form of {s_format}.'
}

event_int_out_format_zh = {
    0:['"(事件触发词,事件类型,[(事件论元,论元角色)])"', event_convert_target0],
    1:['"事件触发词是,事件类型是\n\t事件论元是,论元角色是\n"', event_convert_target1],
    2:['"事件类型：事件触发词\n\t论元角色：事件论元\n"', event_convert_target2],
    3:["JSON字符串[{'event_trigger':'', 'event_type':'', 'arguments':['argument':'', 'role':'']}, ]", event_convert_target3],
}

event_int_out_format_en = {
    0:['(Type, Trigger, Role1#Name; Role2#Name)', event_convert_target0_en],
    1:["{'Type':'', 'Trigger':'','Arguments':'Role1#Name; Role2#Name'}"]
}


def clean(s):
    return s.strip()


def replace_nomiss(s):
    return s.replace(prefix, "")


def ee_post_process0_en(result):    
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
    
    matches = re.findall(EE_triple0_en, result)
    new_record = []
    for m in matches:
        spt = m.split(",")
        type, trigger, args = clean(spt, 0), clean(spt, 1), clean(spt, 2)
        if trigger == NAN or type ==NAN:
            continue
        args_spt = args.split(";")
        arg_list = []
        for it in args_spt:
            it = it.strip().split("#")
            if len(it) != 2:
                continue
            role = it[0].strip()
            name = it[1].strip()
            if role == NAN or name == NAN:
                continue
            arg_list.append([role, name])
        new_record.append([type, trigger, arg_list])
    return new_record



def ee_post_process0_zh(result):    
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
    
    matches = re.findall(EE_triple0_en, result)
    new_record = []
    for m in matches:
        index = m.find('[')
        if index == -1:
            continue
        spt = m[:index-1].split(',')
        type, trigger = clean(spt, 0), clean(spt, 1)
        if trigger == NAN or type ==NAN:
            continue
        arguments = m[index+1:-2]
        arguments_matches = re.findall(EE_triple0_arg_zh, arguments)
        arg_list = []
        for it in arguments_matches:
            spt_ = it.split(",")
            arg, role = clean(spt_, 0), clean(spt_, 1) 
            if role == NAN or arg == NAN:
                continue
            arg_list.append([role, arg])
        new_record.append([type, trigger, arg_list])
    return new_record

'''
def ee_post_process1_zh(result):    
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
    
    spt = result.split('\n')
    evt_dict = {}
    pre, pre_list = NAN, []
    for it in spt:
        if '\t' not in it:
            if len(pre_list) != 0 and pre != NAN:
                evt_dict[pre] = pre_list
            pre = it
            pre_list = []
        else:
            pre_list.append(it)
        
    new_record = []
    for m in matches:
        index = m.find('[')
        if index == -1:
            continue
        spt = m[:index-1].split(',')
        type, trigger = clean(spt, 0), clean(spt, 1)
        if trigger == NAN or type ==NAN:
            continue
        arguments = m[index+1:-2]
        arguments_matches = re.findall(EE_triple0_arg_zh, arguments)
        arg_list = []
        for it in arguments_matches:
            spt_ = it.split(",")
            arg, role = clean(spt_, 0), clean(spt_, 1) 
            if role == NAN or arg == NAN:
                continue
            arg_list.append([role, arg])
        new_record.append([type, trigger, arg_list])
    return new_record
'''


def ee_post_process_en(output):
    clean_output = clean(output)
    if clean_output == NAN:
        return []
    if '(' in output and ')' in output and ',' in output:
        kg = ee_post_process0_en(output)
    #elif "Type" in output and "Trigger" in output and "Arguments" in output and "'" in output and '{' in output and '}' in output and ':' in output:
        #kg = ee_post_process0_en(output)
    else:
        kg = None
    return kg


def ee_post_process_zh(output):
    clean_output = clean(output)
    if clean_output == NAN:
        return []
    if '(' in output and ')' in output and ',' in output and '[' in output and ']' in output:
        kg = ee_post_process0_zh(output)
    else:
        kg = None
    return kg


