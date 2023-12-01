import sys
sys.path.append("./")
import argparse
import json
import os
import random
from typing import Dict
random.seed(42)

from convert.sampler import Sampler
from convert.converter import NERConverter, REConverter, EEAConverter, EETConverter, EEConverter
from utils import stable_hash


def split_by_num(schema_num, task, converter, rand1, rand2, schema1, schema2=""):
    if schema_num <= 0:
        if task == 'EE':
            sinstruct, _ = converter.convert([], rand1, rand2, s_schema1=schema1)
        elif task == 'EEA':
            sinstruct, _ = converter.convert([], rand1, rand2, s_schema1=schema1, s_schema2=schema2)
        elif task == 'EET':
            sinstruct, _ = converter.convert([], rand1, rand2, s_schema1=schema1)
        elif task == 'RE':
            sinstruct, _ = converter.convert([], rand1, rand2, s_schema1=schema1)
        elif task == 'NER':
            sinstruct, _ = converter.convert([], rand1, rand2, s_schema1=schema1)
        else:
            raise KeyError
        sinstruct = [sinstruct, ]
    else:
        sinstruct = []
        if task == 'EE' or task == 'EEA':
            schema_key = list(schema1.keys())
            for i in range(0, len(schema1), schema_num):
                tmp_schema = {}
                tmp_key = schema_key[i:i+schema_num]
                for key in tmp_key:
                    tmp_schema[key] = schema1[key]
                sin, _ = converter.convert([], rand1, rand2, s_schema1=tmp_schema)
                sinstruct.append(sin)
        else:
            for i in range(0, len(schema1), schema_num):
                tmp_schema = schema1[i:i+schema_num]
                sin, _ = converter.convert([], rand1, rand2, s_schema1=tmp_schema)
                sinstruct.append(sin)
    return sinstruct

        

def convert_ie( 
        sample: int, 
        task: str, 
        schema_num: int,
        neg_sampler,
        converter,
    ):
    if sample == -1:           # 从4种指令和4种输出格式(共16种)中随机采样其中一种
        rand1 = random.randint(0,19)
        rand2 = random.randint(0,3)
    else:                      # 使用sample指定的指令和数据格式
        rand1 = sample
        rand2 = sample
    if task == 'EE':
        sinstruct = split_by_num(schema_num, task, converter, rand1, rand2, neg_sampler.type_role_dict)
    elif task == 'EEA':
        sinstruct = split_by_num(schema_num, task, converter, rand1, rand2, neg_sampler.type_role_dict, "")
    elif task == 'EET':
        sinstruct = split_by_num(schema_num, task, converter, rand1, rand2, list(neg_sampler.type_list))
    elif task == 'RE':
        sinstruct = split_by_num(schema_num, task, converter, rand1, rand2, list(neg_sampler.role_list))
    elif task == 'NER':
        sinstruct = split_by_num(schema_num, task, converter, rand1, rand2, list(neg_sampler.type_list))
    else:
        raise KeyError  
    return sinstruct



def process(
        src_path, 
        tgt_path, 
        schema_path, 
        language='zh', 
        task='RE', 
        sample=-1,
        schema_num=-1,
    ):
    if os.path.exists(schema_path):         # 加载该数据集的schema, schema_path文件内容参见utils.py FullSampler.read_from_file
        neg_sampler = Sampler.read_from_file(schema_path, negative=-1)
    else:                                   # 未指定schema_path, 则从数据集中统计得到schema
        raise FileNotFoundError

    if task == 'EE':
        converter = EEConverter(language, NAN='NAN', prefix='')
    elif task == 'RE':
        converter = REConverter(language, NAN='NAN', prefix='')
    elif task == 'NER':
        converter = NERConverter(language, NAN='NAN', prefix='')
    elif task == 'EET':
        converter = EETConverter(language, NAN='NAN', prefix='')
    elif task == 'EEA':
        converter = EEAConverter(language, NAN='NAN', prefix='')
    else:
        raise KeyError
    
    writer = open(tgt_path, "w", encoding="utf-8")
    with open(src_path, "r", encoding="utf-8") as reader:
        for line in reader:
            record = json.loads(line)
            sinstruct = convert_ie(
                sample, 
                task, 
                schema_num,
                neg_sampler,
                converter,
            )
            iid = stable_hash(record['input'])
            for i in range(0, len(sinstruct)):
                new_record = {'id': iid, 'split':i, 'instruction': sinstruct[i], 'input': record['input']}
                writer.write(json.dumps(new_record, ensure_ascii=False)+"\n")


'''
python kg2instruction/convert_test.py \
  --src_path data/EE/sample.json \
  --tgt_path data/test/processed.json \
  --schema_path data/EE/schema.json \
  --language zh \
  --task EE \
  --sample 0 \
  --schema_num 2
'''

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--src_path", type=str, default="data/NER/sample.json")
    parse.add_argument("--tgt_path", type=str, default="data/NER/processed.json")
    parse.add_argument("--schema_path", type=str, default='data/NER/schema.json')
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en'], help="不同语言使用的template及转换脚本不同")
    parse.add_argument("--task", type=str, default="NER", choices=['RE', 'NER', 'EE', 'EET', 'EEA'])
    parse.add_argument("--sample", type=int, default=0, help="若为-1, 则从4种指令和4种输出格式中随机采样其中一种, 否则即为指定的指令格式, -1<=sample<=3")
    parse.add_argument("--schema_num", type=int, default=-1, help="若为-1, 则选取所有schema, 否则即为指定的schema数量, 并切分成多份")


    options = parse.parse_args()
    options = vars(options)
    process(**options)

    