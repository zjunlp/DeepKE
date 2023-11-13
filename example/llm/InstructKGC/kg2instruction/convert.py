import sys
sys.path.append("./")
import argparse
import json
import os
from collections import defaultdict
import random
from typing import Dict
random.seed(42)

from convert.sampler import Sampler, get_positive_type_role
from convert.random_sort import rel_sort, ent_sort, ee_sort, eet_sort
from convert.converter import NERConverter, REConverter, EEAConverter, EETConverter, EEConverter
from utils import stable_hash


def get_schema(src_path, task):    # 从数据集中统计类型列表形成schema
    type_set = set()
    role_set = set()
    type_role_dict = defaultdict(set)
    with open(src_path, "r", encoding="utf-8") as reader:
        for line in reader:
            record = json.loads(line)
            if task == 'EE':
                for event in record['event']:
                    type_set.add(event['event_type'])
                    for arg in event['arguments']:
                        role_set.add(arg['role'])
                        type_role_dict[event['event_type']].add(arg['role'])
            elif task == 'RE':
                for rel in record["relation"]:
                    role_set.add(rel["relation"])
            elif task == 'NER':
                for ent in record["entity"]:
                    type_set.add(ent["entity_type"])
            else:
                raise KeyError
    return list(type_set), list(role_set), type_role_dict




def convert_ie(
        record:Dict, 
        sample:int, 
        task:str, 
        neg_sampler,
        converter,
        neg_ratio:0.1,
        input_text='input',
        random_sort=True,
    ):
    if sample == -1:           # 从4种指令和4种输出格式(共16种)中随机采样其中一种
        rand1 = random.randint(0,19)
        rand2 = random.randint(0,3)
    else:                      # 使用sample指定的指令和数据格式
        rand1 = sample
        rand2 = sample

    neg = False
    if neg_ratio > 0:
        rand3 = random.random() 
        if rand3 < neg_ratio:
            neg = True

    if task == 'EE':
        if neg:      # all表示指定需要抽取的类型是全部schema, 而非仅出现在标签中的类型
            record['event'] = neg_sampler.negative_sample(record['event'], 'EE')
        if random_sort:
            record['event'], type_role_dict = ee_sort(record[input_text], record['event'])
        else:
            type_role_dict = get_positive_type_role(record['event'], 'EE')[2]
            type_role_dict = {k: sorted(list(v)) for k, v in sorted(type_role_dict.items())}
        sinstruct, output_text = converter.convert(record['event'], rand1, rand2, s_schema1=type_role_dict)
    elif task == 'EEA':
        if neg:      # all表示指定需要抽取的类型是全部schema, 而非仅出现在标签中的类型
            record['event'] = neg_sampler.negative_sample(record['event'], 'EEA')
        if random_sort:
            record['event'], type_role_dict = ee_sort(record[input_text], record['event'])
        else:
            type_role_dict = get_positive_type_role(record['event'], 'EE')[2]
            type_role_dict = {k: sorted(list(v)) for k, v in sorted(type_role_dict.items())}
        schema2 = set()
        for event in record['event']:
            schema2.add((event['event_type'], event['event_trigger']))
        schema2_dict = [{'event_type': e[0], 'event_trigger': e[1]} for e in schema2]
        sinstruct, output_text = converter.convert(record['event'], rand1, rand2, s_schema1=type_role_dict, s_schema2=schema2_dict)
    elif task == 'EET':
        if neg:       # all表示指定需要抽取的类型是全部schema, 而非仅出现在标签中的类型
            record['event'] = neg_sampler.negative_sample(record['event'], 'EET')
        if random_sort:
            record['event'], event_type_list = eet_sort(record[input_text], record['event'])
        else:
            event_type_list = list(get_positive_type_role(record['event'], 'EET')[0])
            event_type_list = sorted(event_type_list)
        sinstruct, output_text = converter.convert(record['event'], rand1, rand2, s_schema1=event_type_list)
    elif task == 'RE':
        if neg:
            record['relation'] = neg_sampler.negative_sample(record['relation'], 'RE')
        if random_sort:
            record['relation'], rels_type = rel_sort(record[input_text], record['relation'])    # 按关系、头实体、尾实体随机排序
        else:
            rels_type = list(get_positive_type_role(record['relation'], 'RE')[1])
            rels_type = sorted(rels_type)
        sinstruct, output_text = converter.convert(record['relation'], rand1, rand2, s_schema1=rels_type)
    elif task == 'NER':
        if neg:
            record['entity'] = neg_sampler.negative_sample(record['entity'], 'NER')
        if random_sort:
            record['entity'], ents_type = ent_sort(record[input_text], record['entity'])      # 按实体类型、实体随机排序
        else:
            ents_type = list(get_positive_type_role(record['entity'], 'NER')[0])
            ents_type = sorted(ents_type)
        sinstruct, output_text = converter.convert(record['entity'], rand1, rand2, s_schema1=ents_type)
    else:
        raise KeyError
    return sinstruct, output_text





def process(
        src_path, 
        tgt_path, 
        schema_path, 
        language='zh', 
        task='RE', 
        sample=-1,
        neg_ratio=0.1,
        neg_schema=0.8,
        random_sort=True,
    ):
    if os.path.exists(schema_path):         # 加载该数据集的schema, schema_path文件内容参见utils.py FullSampler.read_from_file
        neg_sampler = Sampler.read_from_file(schema_path, negative=-1)
    else:                                   # 未指定schema_path, 则从数据集中统计得到schema
        type_list, role_list, type_role_dict = get_schema(src_path, task)
        neg_sampler = Sampler(type_list, role_list, type_role_dict, negative=-1)

    if task == 'EE':
        converter = EEConverter(language, NAN='NAN', prefix='')
        neg_sampler.set_negative(max(1, round(neg_schema*len(neg_sampler.type_role_dict))))
    elif task == 'RE':
        converter = REConverter(language, NAN='NAN', prefix='')
        neg_sampler.set_negative(max(1, round(neg_schema*len(neg_sampler.role_list))))
    elif task == 'NER':
        converter = NERConverter(language, NAN='NAN', prefix='')
        neg_sampler.set_negative(max(1, round(neg_schema*len(neg_sampler.type_list))))
    elif task == 'EET':
        converter = EETConverter(language, NAN='NAN', prefix='')
        neg_sampler.set_negative(max(1, round(neg_schema*len(neg_sampler.type_list))))
    elif task == 'EEA':
        converter = EEAConverter(language, NAN='NAN', prefix='')
        neg_sampler.set_negative(max(1, round(neg_schema*len(neg_sampler.role_list))))
    else:
        raise KeyError
    
    writer = open(tgt_path, "w", encoding="utf-8")
    with open(src_path, "r", encoding="utf-8") as reader:
        for line in reader:
            record = json.loads(line)
            sinstruct, output_text = convert_ie(
                record, 
                sample, 
                task, 
                neg_sampler,
                converter,
                neg_ratio=neg_ratio,
                input_text='input',
                random_sort=random_sort,
            )
            new_record = {'id': stable_hash(record['input']),'instruction': sinstruct, 'input': record['input'], 'output': output_text}
            writer.write(json.dumps(new_record, ensure_ascii=False)+"\n")



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
    python kg2instruction/convert.py \
        --src_path data/NER/sample.json \
        --tgt_path data/NER/processed.json \
        --schema_path data/NER/schema.json \
        --language zh \
        --task NER \
        --sample -1 \
        --neg_ratio 1 \
        --neg_schema 1 \
        --random_sort 
    '''

    parse = argparse.ArgumentParser()
    parse.add_argument("--src_path", type=str, default="data/NER/sample.json")
    parse.add_argument("--tgt_path", type=str, default="data/NER/processed.json")
    parse.add_argument("--schema_path", type=str, default='data/NER/schema.json')
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en'], help="不同语言使用的template及转换脚本不同")
    parse.add_argument("--task", type=str, default="NER", choices=['RE', 'NER', 'EE', 'EET', 'EEA'])
    parse.add_argument("--sample", type=int, default=0, help="若为-1, 则从20种指令和4种输出格式中随机采样其中一种, 否则即为指定的指令格式, -1<=sample<20")
    parse.add_argument("--neg_ratio", type=float, default=1, help="0~1之间的小数, 表示所有样本的负采样比例, <=0表示不负采样")
    parse.add_argument("--neg_schema", type=float, default=1, help="0~1之间的小数, 表示从schema中负采样的比例, <=0表示不负采样")
    parse.add_argument("--random_sort", action="store_true", help="是否对指令中的schema列表进行随机排序, 默认不进行随机排序")    
    options = parse.parse_args()
    options = vars(options)
    process(**options)

    