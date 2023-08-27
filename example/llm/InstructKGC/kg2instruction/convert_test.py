import sys
sys.path.append("./")
import argparse
import json
import os
import random
random.seed(42)
from utils import FullSampler
from ner_template import entity_template_zh, entity_int_out_format_zh, entity_template_en, entity_int_out_format_en
from re_template import relation_template_zh, relation_int_out_format_zh, relation_template_en, relation_int_out_format_en
from ee_template import event_template_zh, event_int_out_format_zh, event_template_en, event_int_out_format_en



def process(src_path, tgt_path, schema_path, language='zh', task='RE', sample=-1, all=True):
    if language == 'zh':
        event_template, event_int_out_format = event_template_zh, event_int_out_format_zh 
        relation_template, relation_int_out_format = relation_template_zh, relation_int_out_format_zh 
        entity_template, entity_int_out_format = entity_template_zh, entity_int_out_format_zh 
    else:
        event_template, event_int_out_format = event_template_en, event_int_out_format_en
        relation_template, relation_int_out_format = relation_template_en, relation_int_out_format_en
        entity_template, entity_int_out_format = entity_template_en, entity_int_out_format_en

    if os.path.exists(schema_path):
        neg_sampler = FullSampler.read_from_file(schema_path)      # 加载该数据集的schema, schema_path文件内容参见utils.py FullSampler.read_from_file
    else:
        raise FileNotFoundError

    cnt = 0
    writer = open(tgt_path, "w", encoding="utf-8")
    with open(src_path, "r", encoding="utf-8") as reader:
        for line in reader:
            record = json.loads(line)
            if sample == -1:           # 从4种指令和4种输出格式(共16种)中随机采样其中一种
                rand1 = random.randint(0,3)
                rand2 = random.randint(0,3)
            else:                      # 使用sample指定的指令和数据格式
                rand1 = sample
                rand2 = sample
            if task == 'EE':
                event_type= list(neg_sampler.type_list)
                role = list(neg_sampler.role_list)
                output_template = event_int_out_format[rand2]
                sinstruct = event_template[rand1].format(s_format=output_template[0], s_schema1=event_type, s_schema2=role)
            elif task == 'RE':
                rels_type = list(neg_sampler.role_list)
                output_template = relation_int_out_format[rand2]
                sinstruct = relation_template[rand1].format(s_format=output_template[0], s_schema=list(rels_type))
            elif task == 'NER':
                ents_type = list(neg_sampler.type_list)
                output_template = entity_int_out_format[rand2]
                sinstruct = entity_template[rand1].format(s_format=output_template[0], s_schema=list(ents_type))
            else:
                raise KeyError

            record2 = {'id': cnt,'instruction': sinstruct, 'input': record['input']}
            writer.write(json.dumps(record2, ensure_ascii=False)+"\n")
            cnt += 1




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--src_path", type=str, default="data/NER/sample.json")
    parse.add_argument("--tgt_path", type=str, default="data/NER/processed.json")
    parse.add_argument("--schema_path", type=str, default='data/NER/schema.json')
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en'], help="不同语言使用的template及转换脚本不同")
    parse.add_argument("--task", type=str, default="NER", choices=['RE', 'NER', 'EE'])
    parse.add_argument("--sample", type=int, default=0, help="若为-1, 则从4种指令和4种输出格式中随机采样其中一种, 否则即为指定的指令格式, -1<=sample<=3")
    
    options = parse.parse_args()
    options = vars(options)
    process(**options)

    
