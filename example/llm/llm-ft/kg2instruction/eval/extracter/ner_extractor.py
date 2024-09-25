import re


NER_triple0 = re.compile('(\([^\(\)]*\)?)')
NER_triple3 = re.compile('(\{[^\{\}]*\}?)')
NER_triple3_entity = re.compile("'entity':[\s]?'([^\'\']*?)',")
NER_triple3_type = re.compile("'entity_type':[\s]?'([^\'\']*?)',")



class NERExtractor:
    def __init__(self, language="zh", NAN="NAN", prefix = "输入中包含的实体是：\n", Reject="No entity found."):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix
        self.Reject = Reject


    def clean(self, s):
        return s.strip()

    def replace_nomiss(self, s):
        return s.replace(self.prefix, "")


    def ner_post_process0(self, text, result):    
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
        
        matches = re.findall(NER_triple0, result)
        new_record = []
        for m in matches:
            spt = m.split(",")
            entity, type = clean(spt, 0), clean(spt, -1)
            if entity == "" or type == "":
                continue
            if entity == self.NAN or type == self.NAN:
                continue
            new_record.append((entity, type))

        return new_record



    def ner_post_process1_zh(self, text, result):        
        def clean(spt, i):
            try:
                s = spt[i].strip()
            except IndexError:
                s = ""
            if i == 0 and len(s) != 0:
                s = s.replace("实体是", "")
            if i == -1 and len(s) != 0:
                s = s.replace("实体类型是", "")
            return s

        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            spt = m.split(",")
            entity = clean(spt, 0)
            type = clean(spt, -1)
            if entity == "" or type == "":
                continue
            if entity == self.NAN or type == self.NAN:
                continue
            new_record.append((entity, type))

        return new_record
    

    def ner_post_process1_en(self, text, result):        
        def clean(spt, i):
            try:
                s = spt[i].strip()
            except IndexError:
                s = ""
            if i == 0 and len(s) != 0:
                s = s.replace("Entity is", "")
            if i == -1 and len(s) != 0:
                s = s.replace("Entity Type is", "")
            return s

        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            spt = m.split(",")
            entity = clean(spt, 0)
            type = clean(spt, -1)
            if entity == "" or type == "":
                continue
            if entity == self.NAN or type == self.NAN:
                continue
            new_record.append((entity, type))

        return new_record

    
    def ner_post_process2(self, text, result):        
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
            type, entity = clean(spt[0]), clean(spt[1])
            if entity == "" or type == "":
                continue
            if entity == self.NAN or type == self.NAN:
                continue
            new_record.append((entity, type))

        return new_record


    def ner_post_process3(self, text, result):        
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
            entity = get_span(m, NER_triple3_entity)
            type = get_span(m, NER_triple3_type)
            if entity == "" or type == "":
                continue
            if entity == self.NAN or type == self.NAN:
                continue
            new_record.append((entity, type))
        return new_record
    
    

    def extract(self, text, output):
        clean_output = self.clean(output).lower()
        if clean_output == self.Reject.lower():
            return []
        clean_output = clean_output.replace("\n", "")
        clean_output = clean_output.replace("nan", "")
        if clean_output == "" or "no ent" in clean_output or clean_output.startswith("sorry"):
            return []
        if self.language == "zh":
            return self.ner_post_process_zh(text, output)
        else:
            return self.ner_post_process_en(text, output)


    def ner_post_process_zh(self, text, output):
        if 'entity' in output and 'entity_type' in output and "'" in output and '{' in output and '}' in output and ':' in output:
            kg = self.ner_post_process3(text, output)
        elif '实体是' in output and '实体类型是' in output:
            kg = self.ner_post_process1_zh(text, output)
        elif '(' in output and ')' in output and ',' in output:
            kg = self.ner_post_process0(text, output)
        elif "：" in output:
            kg = self.ner_post_process2(text, output)
        else:
            kg = None
        return kg


    def ner_post_process_en(self, text, output):
        if 'entity' in output and 'entity_type' in output and "'" in output and '{' in output and '}' in output and ':' in output:
            kg = self.ner_post_process3(text, output)
        elif 'Entity is' in output and 'Entity Type is' in output:
            kg = self.ner_post_process1_en(text, output)
        elif '(' in output and ')' in output and ',' in output:
            kg = self.ner_post_process0(text, output)
        elif "：" in output:
            kg = self.ner_post_process2(text, output)
        else:
            kg = None
        return kg
    

    