import sys
sys.path.append('./')
import re


RE_triple0 = re.compile('(\([^\(\)]*\)?)')
RE_triple3 = re.compile('(\{[^\{\}]*\}?)')
RE_triple3_head = re.compile("'head':[\s]?'([^\'\']*?)',")
RE_triple3_rel = re.compile("'relation':[\s]?'([^\'\']*?)',")
RE_triple3_tail = re.compile("'tail':[\s]?'([^\'\']*?)'}")



class REExtractor:
    def __init__(self, language="zh", NAN="NAN", prefix="输入中包含的关系三元组是：\n", Reject=["No relation triples found.", "No relation found."]):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix
        self.Reject = [it.lower() for it in Reject]

    def clean(self, s):
        return s.strip()

    def replace_nomiss(self, s):
        return s.replace(self.prefix, "")
    
    
    def rte_post_process0(self, text, result):  
        def merge_spt(text, spt):
            for i in range(1, len(spt)):
                merge = ','.join(spt[:i])
                index = text.find(merge)
                if index == -1:
                    return ','.join(spt[:i-1]), spt[i-1], ','.join(spt[i:])
            return spt[0], spt[1], ','.join(spt[2:])
      
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
        
        rst = self.replace_nomiss(result)
        matches = re.findall(RE_triple0, rst)
        new_record = []
        for m in matches:
            spt = m.split(",")
            spt[0], spt[-1] = clean(spt, 0), clean(spt, -1)
            if len(spt) > 3:
                head, rel, tail = merge_spt(text, spt)
                print("rte_post_process0", f"[{head} - {rel} - {tail}]")
            elif len(spt) == 3:
                head, rel, tail = spt[0], spt[1], spt[-1]
            else:
                continue
            if head == "" or rel == "" or tail == "":
                continue
            if head == self.NAN or tail == self.NAN:
                continue
            new_record.append((head, rel, tail))
        return new_record


    def rte_post_process1_zh(self, text, result):  
        def other_process(m):
            try:
                head = m.split(",关系是")[0].replace("头实体是", "").strip()
                rel = m.split(",关系是")[-1].split(",尾实体是")[0].strip()
                tail = m.split(",尾实体是")[-1].strip()
            except IndexError:
                return "", "", ""
            return head, rel, tail    
          
        def clean(spt, i):
            try:
                s = spt[i].strip()
            except IndexError:
                s = ""
            if i == 0 and len(s) != 0:
                s = s.replace("头实体是", "")
            if i == 1 and len(s) != 0:
                s = s.replace("关系是", "")
            if i == -1 and len(s) != 0:
                s = s.replace("尾实体是", "")
            return s

        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            spt = m.split(",")
            if len(spt) > 3:
                head, rel, tail = other_process(m)
                print("rte_post_process1_zh", f"[{head} - {rel} - {tail}]")
            else:
                head = clean(spt, 0)
                rel = clean(spt, 1)
                tail = clean(spt, -1)
            if head == "" or rel == "" or tail == "":
                continue
            if head == self.NAN or tail == self.NAN:
                continue
            new_record.append((head, rel, tail))

        return new_record
    

    def rte_post_process1_en(self, text, result): 
        def other_process(m):
            try:
                head = m.split(",Relation is")[0].replace("Subject is", "").strip()
                rel = m.split(",Relation is")[-1].split(",Object is")[0].strip()
                tail = m.split(",Object is")[-1].strip()
            except IndexError:
                return "", "", ""
            return head, rel, tail
               
        def clean(spt, i):
            try:
                s = spt[i].strip()
            except IndexError:
                s = ""         
            if i == 0 and len(s) != 0:
                s = s.replace("Subject is", "")
            if i == 1 and len(s) != 0:
                s = s.replace("Relation is", "")
            if i == -1 and len(s) != 0:
                s = s.replace("Object is", "")
            return s

        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            spt = m.split(",")
            if len(spt) > 3:
                head, rel, tail = other_process(m)
                print("rte_post_process1_zh", f"[{head} - {rel} - {tail}]")
            else:
                head = clean(spt, 0)
                rel = clean(spt, 1)
                tail = clean(spt, -1)
            if head == "" or rel == "" or tail == "":
                continue
            if head == self.NAN or tail == self.NAN:
                continue
            new_record.append((head, rel, tail))

        return new_record


    def rte_post_process2(self, text, result):  
        def merge_spt(text, spt):
            for i in range(1, len(spt)):
                merge = ','.join(spt[:i])
                index = text.find(merge)
                if index == -1:
                    return ','.join(spt[:i-1]), ','.join(spt[i-1:])
            return spt[0], spt[-1]
              
        def clean(spt):
            spt = spt.strip()
            return spt  
        
        rst = self.replace_nomiss(result)
        matches = rst.split("\n")
        new_record = []
        for m in matches:
            if m.strip() == self.NAN:
                continue
            index1 = m.find("：")
            rel = clean(m[:index1])
            head_tail = clean(m[index1+1:])
            index2 = head_tail.find(",")
            spt = head_tail.split(",")
            if len(spt) > 2:
                head, tail = merge_spt(text, spt)
                print("rte_post_process2", f"[{head} - {rel} - {tail}]")
            else:
                head = clean(spt[0])
                tail = clean(spt[-1])
            if head == "" or rel == "" or tail == "":
                continue
            if head == self.NAN or tail == self.NAN:
                continue
            new_record.append((head, rel, tail))

        return new_record


    def rte_post_process3(self, text, result):        
        def get_span(text, template):
            match = re.findall(template, text)
            try: 
                match = match[0]
            except IndexError:
                match = ""
            return match.strip()

        rst = self.replace_nomiss(result)
        matches = re.findall(RE_triple3, rst)
        new_record = []
        for m in matches:
            head = get_span(m, RE_triple3_head)
            rel = get_span(m, RE_triple3_rel)
            tail = get_span(m, RE_triple3_tail)
            if head == "" or rel == "" or tail == "":
                continue
            if head == self.NAN or tail == self.NAN:
                continue
            new_record.append((head, rel, tail))

        return new_record


    def extract(self, text, output):
        clean_output = self.clean(output).lower()
        if clean_output in self.Reject:
            return []
        clean_output = clean_output.replace("nan", "")
        clean_output = clean_output.replace("\n", "")
        if clean_output == "" or clean_output.startswith("sorry") or "not contain" in clean_output or "no relation" in clean_output:
            return []
        if self.language == "zh":
            return self.re_post_process_zh(text, output)
        else:
            return self.re_post_process_en(text, output)


    def re_post_process_zh(self, text, output):
        if 'head' in output and 'relation' in output and 'tail' in output and "'" in output and '{' in output and '}' in output and ':' in output:
            kg = self.rte_post_process3(text, output)
        elif '头实体是' in output and '尾实体是' in output and '关系是' in output:
            kg = self.rte_post_process1_zh(text, output)
        elif '(' in output and ')' in output and ',' in output:
            kg = self.rte_post_process0(text, output)
        elif "：" in output and "," in output:
            kg = self.rte_post_process2(text, output)
        else:
            kg = None
        return kg


    def re_post_process_en(self, text, output):
        if "head" in output and "relation" in output and "tail" in output and "'" in output and '{' in output and '}' in output and ':' in output:
            kg = self.rte_post_process3(text, output)
        elif 'Subject is' in output and 'Object is' in output and 'Relation is' in output:
            kg = self.rte_post_process1_en(text, output)
        elif '(' in output and ')' in output and ',' in output:
            kg = self.rte_post_process0(text, output)
        elif "：" in output and "," in output:
            kg = self.rte_post_process2(text, output)
        else:
            kg = None
        return kg

