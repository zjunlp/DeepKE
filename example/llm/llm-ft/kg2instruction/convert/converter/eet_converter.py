import json

class EETConverter:
    def __init__(self, language="zh", NAN="", prefix="", template_path='configs/eet_template.json'):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix

        template = json.load(open(template_path, 'r'))
        self.event_template = template[language]['template']
        self.event_open_template = template[language]['open_template']

        self.event_int_out_format_zh = {
            0:['(事件触发词,事件类型)\n', self.event_convert_target0],
            1:['事件触发词是,事件类型是\n', self.event_convert_target1],
            2:['事件类型：事件触发词\n', self.event_convert_target2],
            3:["{'event_trigger':'', 'event_type':''}\n", self.event_convert_target3],
        }
        self.event_int_out_format_en = {
            0:['(Event Trigger,Event Type)\n', self.event_convert_target0],
            1:['Event Trigger is,Event Type is\n', self.event_convert_target1_en],
            2:['Event Type：Event Trigger\n', self.event_convert_target2],
            3:["{'event_trigger':'', 'event_type':''}\n", self.event_convert_target3],
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
            output_text.append('(' + ','.join([event_trigger, event_type]) + ')')
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
            output_text.append( f"事件触发词是{event_trigger},事件类型是{event_type}")
        output_text = '\n'.join(output_text)
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
            output_text.append( f"Event Trigger is {event_trigger},Event Type is {event_type}")
        output_text = '\n'.join(output_text)
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
            output_text.append( f"{event_type}：{event_trigger}")
        output_text = '\n'.join(output_text)
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
            output_text.append("{'event_trigger':'%s', 'event_type':'%s'}" % (event_trigger, event_type))
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



