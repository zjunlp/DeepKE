import json

class REConverter:
    def __init__(self, language = "zh", NAN="NAN", prefix="", template_path='configs/re_template.json'):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix

        template = json.load(open(template_path, 'r'))
        self.relation_template = template[language]['template']
        self.relation_open_template = template[language]['open_template']

        self.relation_int_out_format_zh = {
            0:['(头实体,关系,尾实体)\n', self.relation_convert_target0],
            1:['头实体是,关系是,尾实体是\n', self.relation_convert_target1],
            2:['关系：头实体,尾实体\n', self.relation_convert_target2],
            3:["{'head':'', 'relation':'', 'tail':''}\n", self.relation_convert_target3],
        }
        self.relation_int_out_format_en = {
            0:['(Subject,Relation,Object)\n', self.relation_convert_target0],
            1:['Subject is,Relation is,Object is\n', self.relation_convert_target1_en],
            2:['Relation：Subject,Object\n', self.relation_convert_target2],
            3:["{'head':'', 'relation':'', 'tail':''}\n", self.relation_convert_target3],
        }
        if language == "zh":
            self.relation_int_out_format = self.relation_int_out_format_zh
        else:
            self.relation_int_out_format = self.relation_int_out_format_en


    def nan(self, s):
        return s


    def relation_convert_target0(self, rels):
        output_text = []
        for rel in rels:
            head = self.nan(rel['head'])
            relation = self.nan(rel['relation'])
            tail = self.nan(rel['tail'])
            if head == "" or relation == "" or tail == "":
                output_text.append(self.NAN)
                continue
            output_text.append('(' + ','.join([head, relation, tail]) + ')')
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 
    
        
    def relation_convert_target1(self, rels):
        output_text = []
        for rel in rels:
            head = self.nan(rel['head'])
            relation = self.nan(rel['relation'])
            tail = self.nan(rel['tail'])
            if head == "" or relation == "" or tail == "":
                output_text.append(self.NAN)
                continue
            output_text.append(f"头实体是{head},关系是{relation},尾实体是{tail}")
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text
    

    def relation_convert_target1_en(self, rels):
        output_text = []
        for rel in rels:
            head = self.nan(rel['head'])
            relation = self.nan(rel['relation'])
            tail = self.nan(rel['tail'])
            if head == "" or relation == "" or tail == "":
                output_text.append(self.NAN)
                continue
            output_text.append(f"Subject is {head},Relation is {relation},Object is {tail}")
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text


    def relation_convert_target2(self, rels):
        output_text = []
        for rel in rels:
            head = self.nan(rel['head'])
            relation = self.nan(rel['relation'])
            tail = self.nan(rel['tail'])
            if head == "" or relation == "" or tail == "":
                output_text.append(self.NAN)
                continue
            output_text.append(f"{relation}：{head},{tail}")
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 


    def relation_convert_target3(self, rels):
        output_text = []
        for rel in rels:
            head = self.nan(rel['head'])
            relation = self.nan(rel['relation'])
            tail = self.nan(rel['tail'])
            if head == "" or relation == "" or tail == "":
                output_text.append(self.NAN)
                continue
            output_text.append(str({"head":head, "relation":relation, "tail":tail}))
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text
    

    def convert(self, record, rand1, rand2, s_schema1="", s_schema2=""):
        output_template = self.relation_int_out_format[rand2]
        output_text = output_template[1](record)
        sinstruct = self.relation_template[str(rand1)].format(s_format=output_template[0], s_schema=s_schema1)
        return sinstruct, output_text
    
    def convert_open(self, record, rand1, rand2, s_schema1="", s_schema2=""):
        output_template = self.relation_int_out_format[rand2]
        output_text = output_template[1](record)
        sinstruct = self.relation_open_template[str(rand1)].format(s_format=output_template[0], s_schema=s_schema1)
        return sinstruct, output_text




