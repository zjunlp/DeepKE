import json

class NERConverter:
    def __init__(self, language = "zh", NAN="NAN", prefix = "", template_path="configs/ner_template.json"):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix

        template = json.load(open(template_path, 'r'))
        self.entity_template = template[language]['template']
        self.entity_open_template = template[language]['open_template']

        self.entity_int_out_format_zh = {
            0:['(实体,实体类型)\n', self.entity_convert_target0],
            1:['实体是,实体类型是\n', self.entity_convert_target1],
            2:['实体类型：实体\n', self.entity_convert_target2],
            3:["{'entity':'', 'entity_type':''}\n", self.entity_convert_target3],
        }
        self.entity_int_out_format_en = {
            0:['(Entity,Entity Type)\n', self.entity_convert_target0],
            1:['Entity is,Entity Type is\n', self.entity_convert_target1_en],
            2:['Entity Type：Entity\n', self.entity_convert_target2],
            3:["{'entity':'', 'entity_type':''}\n", self.entity_convert_target3],
        } 

        if language == 'zh':
            self.entity_int_out_format = self.entity_int_out_format_zh
        else:
            self.entity_int_out_format = self.entity_int_out_format_en

        
    def nan(self, s):
        #if s.strip() == "":
            #return self.NAN
        return s

    def entity_convert_target0(self, entities): 
        output_text = []
        for entity in entities:
            entity_text = self.nan(entity['entity'])
            entity_type = self.nan(entity['entity_type'])
            if entity_type == "":
                continue
            if entity_text == "":
                output_text.append(self.NAN)
                continue
            output_text.append('(' + ','.join([entity_text, entity_type]) + ')')
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text


    def entity_convert_target1(self, entities):
        output_text = []
        for entity in entities:
            entity_text = self.nan(entity['entity'])
            entity_type = self.nan(entity['entity_type'])
            if entity_type == "":
                continue
            if entity_text == "":
                output_text.append(self.NAN)
                continue
            output_text.append(f"实体是{entity_text},实体类型是{entity_type}")
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 
    

    def entity_convert_target1_en(self, entities):
        output_text = []
        for entity in entities:
            entity_text = self.nan(entity['entity'])
            entity_type = self.nan(entity['entity_type'])
            if entity_type == "":
                continue
            if entity_text == "":
                output_text.append(self.NAN)
                continue
            output_text.append(f"Entity is {entity_text},Entity Type is {entity_type}")
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 


    def entity_convert_target2(self, entities):
        output_text = []
        for entity in entities:
            entity_text = self.nan(entity['entity'])
            entity_type = self.nan(entity['entity_type'])
            if entity_type == "":
                continue
            if entity_text == "":
                output_text.append(self.NAN)
                continue
            output_text.append(f"{entity_type}：{entity_text}")
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text 


    def entity_convert_target3(self, entities): 
        output_text = []
        for entity in entities:
            entity_text = self.nan(entity['entity'])
            entity_type = self.nan(entity['entity_type'])
            if entity_type == "":
                continue
            if entity_text == "":
                output_text.append(self.NAN)
                continue
            output_text.append(str({"entity":entity_text, "entity_type":entity_type}))
        output_text = '\n'.join(output_text)
        if len(output_text.replace(self.NAN, '').replace('\n', '').strip()) == 0:
            return self.prefix + self.NAN
        return self.prefix + output_text
    
    
    def convert(self, record, rand1, rand2, s_schema1="", s_schema2=""):
        output_template = self.entity_int_out_format[rand2]
        output_text = output_template[1](record)
        sinstruct = self.entity_template[str(rand1)].format(s_format=output_template[0], s_schema=s_schema1)
        return sinstruct, output_text
    
    def convert_open(self, record, rand1, rand2, s_schema1="", s_schema2=""):
        output_template = self.entity_int_out_format[rand2]
        output_text = output_template[1](record)
        sinstruct = self.entity_open_template[str(rand1)].format(s_format=output_template[0], s_schema=s_schema1)
        return sinstruct, output_text




