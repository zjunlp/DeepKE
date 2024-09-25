import re

EE_triple0 = re.compile('(\([^\(\)]*\)?)')
EE_triple3 = re.compile('(\{[^\{\}]*\}?)')
EE_triple3_trigger = re.compile("'event_trigger':[\s]?'([^\'\']*?)',")
EE_triple3_type = re.compile("'event_type':[\s]?'([^\'\']*?)',")


class EETExtractor:
    def __init__(self, language="zh", NAN="NAN", prefix="输入中包含的事件是：\n", Reject="No event found."):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix
        self.Reject = Reject

    def clean(self, s):
        return s.strip()

    def replace_nomiss(self, s):
        return s.replace(self.prefix, "")
    

    def ee_post_process0(self, text, result):    
        def clean(spt, i):
            try:
                s = spt[i].strip()
            except IndexError:
                s = ""
            if i == 0 and len(s) != 0 and s[0] == '(':
                s = s[1:]
            if i == -1 and len(s) != 0 and s[-1] == ')':
                s = s[:-1]
            return s
        
        matches = re.findall(EE_triple0, result)
        new_record = []
        for m in matches:
            spt = m.split(",")
            event_type, trigger = clean(spt, 0), clean(spt, -1)
            if trigger == self.NAN or event_type == self.NAN:
                continue
            new_record.append((event_type, trigger))
        return new_record



    def ee_post_process1_zh(self, text, result):
        def clean_type(spt, i):
            try:
                s = spt[i].strip()
            except IndexError:
                s = ""
            if i == 0 and len(s) != 0:
                s = s.replace("事件触发词是", "")
            if i == -1 and len(s) != 0:
                s = s.replace("事件类型是", "")
            return s

        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            spt = m.split(",")
            trigger = clean_type(spt, 0)
            event_type = clean_type(spt, -1)
            if trigger == "" or event_type == "":
                continue
            if trigger == self.NAN or event_type == self.NAN:
                continue
            new_record.append((event_type, trigger))
        return new_record
    

    def ee_post_process1_en(self, text, result):
        def clean_type(spt, i):
            try:
                s = spt[i].strip()
            except IndexError:
                s = ""
            if i == 0 and len(s) != 0:
                s = s.replace("Event Trigger is", "")
            if i == -1 and len(s) != 0:
                s = s.replace("Event Type is", "")
            return s

        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            spt = m.split("\n")
            trigger = clean_type(spt, 0)
            event_type = clean_type(spt, -1)
            if trigger == "" or event_type == "":
                continue
            if trigger == self.NAN or event_type == self.NAN:
                continue
            new_record.append((event_type, trigger))
        return new_record
    

    def ee_post_process2(self, text, result):
        def clean(spt):
            spt = spt.strip()
            return spt  

        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            spt = m.split("：")
            event_type = clean(spt[0])
            trigger = clean(spt[-1])
            if trigger == "" or event_type == "":
                continue
            if trigger == self.NAN or event_type == self.NAN:
                continue
            new_record.append((event_type, trigger))
        return new_record


    def ee_post_process3(self, text, result):        
        def get_span(text, template):
            match = re.findall(template, text)
            try: 
                match = match[0]
            except IndexError:
                match = ""
            return match.strip()

        rst = self.replace_nomiss(result)
        matches = re.findall(EE_triple3, rst)
        new_record = []

        for m in matches:
            trigger = get_span(m, EE_triple3_trigger)
            event_type = get_span(m, EE_triple3_type)
            if trigger == "" or event_type == "":
                continue
            if trigger == self.NAN or event_type == self.NAN:
                continue
            new_record.append((event_type, trigger))
        return new_record


    def extract(self, text, output):
        clean_output = self.clean(output).lower()
        if clean_output == self.Reject.lower():
            return []
        clean_output = clean_output.replace("\n", "")
        clean_output = clean_output.replace("nan", "")
        if clean_output == "":
            return []
        if self.language == "zh":
            return self.ee_post_process_zh(text, output)
        else:
            return self.ee_post_process_en(text, output)
        

    def ee_post_process_en(self, text, output):
        if 'event_trigger' in output and 'event_type' in output and "'" in output and '{' in output and '}' in output and ':' in output:
            kg = self.ee_post_process3(text, output)
        elif 'Event Trigger is' in output and 'Event Type is' in output:
            kg = self.ee_post_process1_en(text, output)
        elif '(' in output and ')' in output and ',' in output:
            kg = self.ee_post_process0(text, output)
        elif '：' in output:
            kg = self.ee_post_process2(text, output)
        else:
            kg = None
        return kg


    def ee_post_process_zh(self, text, output):
        if 'event_trigger' in output and 'event_type' in output and "'" in output and '{' in output and '}' in output and ':' in output:
            kg = self.ee_post_process3(text, output)
        elif '事件触发词是' in output and '事件类型是' in output:
            kg = self.ee_post_process1_zh(text, output)
        elif '(' in output and ')' in output and ',' in output:
            kg = self.ee_post_process0(text, output)
        elif '：' in output:
            kg = self.ee_post_process2(text, output)
        else:
            kg = None
        return kg
