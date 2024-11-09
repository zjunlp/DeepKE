import json

class EEConverter:
    def __init__(self, language="zh", NAN="NAN", prefix="", template_path="configs/ee_template.json"):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix
        
        template = json.load(open(template_path, 'r'))
        self.event_template = template[language]['template']
        self.event_open_template = template[language]['open_template']

        self.event_int_out_format_zh = {
            0:['(事件触发词,事件类型,事件论元1#论元角色1;事件论元2#论元角色2)\n', self.event_convert_target0],
            1:['事件触发词是,事件类型是\n事件论元1是,论元角色1是;事件论元2是,论元角色2是\n\n', self.event_convert_target1],
            2:['事件类型：事件触发词\n论元角色1：事件论元1;论元角色2：事件论元2\n\n', self.event_convert_target2],
            3:["{'event_trigger':'', 'event_type':'', 'arguments':({'argument':'', 'role':''},)}\n", self.event_convert_target3],
        }
        self.event_int_out_format_en = {
            0:['(Event Trigger,Event Type,Argument1#Argument Role1;Argument2#Argument Role2)\n', self.event_convert_target0],
            1:['Event Trigger is,Event Type is\nArgument1 is,Argument Role1 is;Argument2 is,Argument Role2 is\n\n', self.event_convert_target1_en],
            2:['Event Type：Event Trigger\nArgument Role1：Argument1;Argument Role2：Argument2\n\n', self.event_convert_target2],
            3:["{'event_trigger':'', 'event_type':'', 'arguments':({'argument':'', 'role':''},)}\n", self.event_convert_target3],
        }
        if language == 'zh':
            self.event_int_out_format = self.event_int_out_format_zh
        else:
            self.event_int_out_format = self.event_int_out_format_en

    def nan(self, s):
        return s

    def event_convert_target0(self, events):
        output_text = []
        for event in events:
            event_trigger = self.nan(event['event_trigger'])
            event_type = self.nan(event['event_type'])
            if event_type == "":
                continue
            if event_trigger == "":
                output_text.append(self.NAN)
                continue
            args = []
            for arg in event['arguments']:
                argument = self.nan(arg['argument'])
                role = self.nan(arg['role'])
                if role == "":
                    continue
                if argument == "":
                    args.append(self.NAN)
                    continue
                args.append(f"{role}#{argument}")
            if len(args) == 0 and self.NAN != "":
                args.append(self.NAN)
            args = ';'.join(args)
            output_text.append('(' + ','.join([event_trigger, event_type, args]) + ')')
        output_text = '\n'.join(output_text) 
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 


    def event_convert_target1(self, events):
        output_text = []
        for event in events:
            event_trigger = self.nan(event['event_trigger'])
            event_type = self.nan(event['event_type'])
            if event_type == "":
                continue
            if event_trigger == "":
                output_text.append(self.NAN)
                continue
            args = []
            for arg in event['arguments']:
                argument = self.nan(arg['argument'])
                role = self.nan(arg['role'])
                if role == "":
                    continue
                if argument == "":
                    args.append(self.NAN)
                    continue
                args.append(f"事件论元是{argument},论元角色是{role}")
            if len(args) == 0 and self.NAN != "":
                args.append(self.NAN)
            args = ';'.join(args)
            output_text.append( f"事件触发词是{event_trigger},事件类型是{event_type}" + "\n" + args)
        output_text = '\n\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 
    

    def event_convert_target1_en(self, events):
        output_text = []
        for event in events:
            event_trigger = self.nan(event['event_trigger'])
            event_type = self.nan(event['event_type'])
            if event_type == "":
                continue
            if event_trigger == "":
                output_text.append(self.NAN)
                continue
            args = []
            for arg in event['arguments']:
                argument = self.nan(arg['argument'])
                role = self.nan(arg['role'])
                if role == "":
                    continue
                if argument == "":
                    args.append(self.NAN)
                    continue
                args.append(f"Argument is {argument},Argument Role is {role}")
            if len(args) == 0 and self.NAN != "":
                args.append(self.NAN)
            args = ';'.join(args)
            output_text.append( f"Event Trigger is {event_trigger},Event Type is {event_type}" + "\n" + args)
        output_text = '\n\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 


    def event_convert_target2(self, events):
        output_text = []
        for event in events:
            event_trigger = self.nan(event['event_trigger'])
            event_type = self.nan(event['event_type'])
            if event_type == "":
                continue
            if event_trigger == "":
                output_text.append(self.NAN)
                continue
            args = []
            for arg in event['arguments']:
                argument = self.nan(arg['argument'])
                role = self.nan(arg['role'])
                if role == "":
                    continue
                if argument == "":
                    args.append(self.NAN)
                    continue
                args.append(f"{role}：{argument}")
            if len(args) == 0 and self.NAN != "":
                args.append(self.NAN)
            args = ';'.join(args)
            output_text.append( f"{event_type}：{event_trigger}" + "\n" + args)
        output_text = '\n\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 


    def event_convert_target3(self, events):
        output_text = []
        for event in events:
            event_trigger = self.nan(event['event_trigger'])
            event_type = self.nan(event['event_type'])
            if event_type == "":
                continue
            if event_trigger == "":
                output_text.append(self.NAN)
                continue
            args = []
            for arg in event['arguments']:
                argument = self.nan(arg['argument'])
                role = self.nan(arg['role'])
                if role == "":
                    continue
                if argument == "":
                    args.append(self.NAN)
                    continue
                args.append("{'argument':'%s', 'role':'%s'}" % (argument, role))
            if len(args) == 0 and self.NAN != "":
                args.append(self.NAN)
            args_str = '(' + ','.join(args) + ')'
            output_text.append("{'event_trigger':'%s', 'event_type':'%s', 'arguments':%s}" % (event_trigger, event_type, args_str))
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text


    def convert(self, record, rand1, rand2, s_schema1="", s_schema2=""):
        output_template = self.event_int_out_format[rand2]
        output_text = output_template[1](record)
        sinstruct = self.event_template[str(rand1)].format(s_format=output_template[0], s_schema=s_schema1)
        return sinstruct, output_text

    def convert_open(self, record, rand1, rand2, s_schema1="", s_schema2=""):
        output_template = self.event_int_out_format[rand2]
        output_text = output_template[1](record)
        sinstruct = self.event_open_template[str(rand1)].format(s_format=output_template[0], s_schema=s_schema1)
        return sinstruct, output_text



