import sys
sys.path.append("./")
import argparse
import json
import os
import random
random.seed(42)
from utils import FullSampler, rel_sort, ent_sort
from ner_template import entity_template_zh, entity_int_out_format_zh, entity_template_en, entity_int_out_format_en
from re_template import relation_template_zh, relation_int_out_format_zh, relation_template_en, relation_int_out_format_en
from ee_template import event_template_zh, event_int_out_format_zh, event_template_en, event_int_out_format_en




def get_schema(src_path, task):    # 从数据集中统计类型列表形成schema
    type_set = set()
    role_set = set()
    with open(src_path, "r", encoding="utf-8") as reader:
        for line in reader:
            record = json.loads(line)
            if task == 'EE':
                for event in record['event']:
                    type_set.add(event['event_type'])
                    for arg in event['arguments']:
                        role_set.add(arg['role'])
            elif task == 'RE':
                for rel in record["relation"]:
                    role_set.add(rel["relation"])
            elif task == 'NER':
                for ent in record["entity"]:
                    type_set.add(ent["entity_type"])
            else:
                raise KeyError
    return list(type_set), list(role_set)




def process(src_path, tgt_path, schema_path, language='zh', task='RE', sample=-1, all=True):
    if language == 'zh':
        event_template, event_int_out_format = event_template_zh, event_int_out_format_zh 
        relation_template, relation_int_out_format = relation_template_zh, relation_int_out_format_zh 
        entity_template, entity_int_out_format = entity_template_zh, entity_int_out_format_zh 
    else:
        event_template, event_int_out_format = event_template_en, event_int_out_format_en
        relation_template, relation_int_out_format = relation_template_en, relation_int_out_format_en
        entity_template, entity_int_out_format = entity_template_en, entity_int_out_format_en

    
    if os.path.exists(schema_path):     # 加载该数据集的schema, schema_path文件内容参见utils.py FullSampler.read_from_file
        neg_sampler = FullSampler.read_from_file(schema_path)
    else:                               # 未指定schema_path, 则从数据集中统计得到schema
        type_list, role_list = get_schema(src_path, task)
        neg_sampler = FullSampler(type_list, role_list)


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
                if all:   # all表示指定需要抽取的类型是全部schema, 而非仅出现在标签中的类型
                    record['event'], event_type_set, role_set = neg_sampler.negative_sample(record['event'], 'EE')
                else:
                    event_type_set = set()
                    role_set = set()
                    for event in record['event']:
                        event_type_set.add(event['event_type'])
                        for arg in event['arguments']:
                            role_set.add(arg['role'])
                    event_type_set = list(event_type_set)
                    role_set = list(role_set)
                output_template = event_int_out_format[rand2]
                output_text = output_template[1](record['event'])
                sinstruct = event_template[rand1].format(s_format=output_template[0], s_schema1=event_type_set, s_schema2=role_set)
            elif task == 'RE':
                if all:
                    record['relation'], rels_type, _ = neg_sampler.negative_sample(record['relation'], 'RE')
                new_rels, rels_type = rel_sort(record['input'], record['relation'])    # 按关系、头实体、尾实体随机排序
                output_template = relation_int_out_format[rand2]
                output_text = output_template[1](new_rels)
                sinstruct = relation_template[rand1].format(s_format=output_template[0], s_schema=list(rels_type))
            elif task == 'NER':
                if all:
                    record['entity'], ents_type, _ = neg_sampler.negative_sample(record['entity'], 'NER')
                new_ents, ents_type = ent_sort(record['input'], record['entity'])      # 按实体类型、实体随机排序
                output_template = entity_int_out_format[rand2]
                output_text = output_template[1](new_ents)
                sinstruct = entity_template[rand1].format(s_format=output_template[0], s_schema=list(ents_type))
            else:
                raise KeyError

            record2 = {'id': cnt,'instruction': sinstruct, 'input': record['input'], 'output': output_text}
            writer.write(json.dumps(record2, ensure_ascii=False)+"\n")
            cnt += 1




if __name__ == "__main__":
    '''
    src_path 和 schema_path具体文件格式参考data目录下的NER、RE、EE(不同任务有所不同)

    NER schema:
    ["人物", "组织机构", "地理位置"]
    []
    {}
    对于下面的例子
    {
        "text": "有心无力的前卫寰岛队只靠高峰扳回一球。", 
        "entity": [{"entity": "前卫寰岛队", "entity_type": "组织机构"}, {"entity": "高峰", "entity_type": "人物"}]
    }
    如果all == True
        instruction = 已知候选的实体类型列表：["人物", "组织机构", "地理位置"]，请你根据实体类型列表，从以下输入中抽取出可能存在的实体。请按照{s_format}的格式回答。
    否则
        instruction = 已知候选的实体类型列表：["人物", "组织机构"]，请你根据实体类型列表，从以下输入中抽取出可能存在的实体。请按照{s_format}的格式回答。
    
    python kg2instruction/convert.py \
        --src_path data/NER/sample.json \
        --tgt_path data/NER/processed.json \
        --schema_path data/NER/schema.json \
        --language zh \
        --task NER \
        --sample 0 \
        --all

    '''

    parse = argparse.ArgumentParser()
    parse.add_argument("--src_path", type=str, default="data/NER/sample.json")
    parse.add_argument("--tgt_path", type=str, default="data/NER/processed.json")
    parse.add_argument("--schema_path", type=str, default='data/NER/schema.json')
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en'], help="不同语言使用的template及转换脚本不同")
    parse.add_argument("--task", type=str, default="NER", choices=['RE', 'NER', 'EE'])
    parse.add_argument("--sample", type=int, default=0, help="若为-1, 则从4种指令和4种输出格式中随机采样其中一种, 否则即为指定的指令格式, -1<=sample<=3")
    parse.add_argument("--all", action='store_true', help="是否将指令中指定的抽取类型列表设置为全部schema")
    
    options = parse.parse_args()
    options = vars(options)
    process(**options)

    