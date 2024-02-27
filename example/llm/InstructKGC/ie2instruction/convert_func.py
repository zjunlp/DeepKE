import sys
sys.path.append("./")
import argparse
import json
import random
random.seed(42)

from convert.utils.instruction import instruction_mapper
from convert.utils.utils import stable_hash, write_to_json
from convert.processer import get_processer
from convert.converter import get_converter


def multischema_split_by_num_test(schemas, split_num=4):
    if len(schemas) < split_num or split_num == -1:
        return [schemas, ]

    negative_length = max(len(schemas) // split_num, 1) * split_num
    total_schemas = []
    for i in range(0, negative_length, split_num):
        total_schemas.append(schemas[i:i+split_num])

    remain_len = max(1, split_num // 2)
    if len(schemas) - negative_length >= remain_len:
        total_schemas.append(schemas[negative_length:])
    else:
        total_schemas[-1].extend(schemas[negative_length:])
    return total_schemas


def multischema_construct_instruction(task, language, schema1, text):
    instruction = {
        "instruction":instruction_mapper[task+language],
        "schema":schema1,
        "input":text,
    }
    return json.dumps(instruction, ensure_ascii=False)  


def get_test_data(datas, processer, options):
    results = []
    for record in datas:
        iid = stable_hash(record['text'])
        task_record = processer.get_task_record(record)  
        schemas = processer.get_schemas(task_record)
        if schemas is None:
            continue
        total_schemas = multischema_split_by_num_test(schemas, options.split_num)  
        for schema in total_schemas:
            sinstruct = multischema_construct_instruction(options.task, options.language, schema, record['text'])
            record2 = {
                'id': iid,
                'task': options.task,
                'source': options.source,
                'instruction': sinstruct, 
            }
            if task_record is not None:
                record2['label'] = json.dumps(task_record, ensure_ascii=False)
            results.append(record2)
    return results


def convert_output(converter, text, schemas, task_record):
    output_texts = []
    if len(schemas) == 0:
        return output_texts
    label_dict = converter.get_label_dict(task_record)
    for schema in schemas:
        output_text = converter.convert(
            text, label_dict, s_schema1=schema
        )
        output_texts.append(output_text)
    return output_texts


def get_train_data(datas, processer, converter, options):
    results = []
    for record in datas:
        total_schemas = processer.negative_sample(record, options.split_num, options.random_sort)        
        task_record = processer.get_task_record(record)
        output_texts = convert_output(converter, record['text'], total_schemas, task_record)    # 按照split_num切分schema和output_text
        for schema, output_text in zip(total_schemas, output_texts):
            sinstruct = multischema_construct_instruction(options.task, options.language, schema, record['text'])
            record2 = {
                'task': options.task,
                'source': options.source,
                'instruction': sinstruct, 
                'output': output_text
            }
            results.append(record2)
    return results


def process(options):
    converter = get_converter(options.task)(options.language, NAN='NAN')
    processer_class = get_processer(options.task)  
    processer = processer_class.read_from_file(    
        processer_class, options.schema_path, negative=-1
    )
    processer.set_negative(options.neg_schema)
    options.source = options.src_path.split('/')[-2]  # 用源路径的最后一个文件夹名作为source
    
    datas = processer.read_data(options.src_path)
    if options.split == 'test':  
        results = get_test_data(datas, processer, options)
    else:
        results = get_train_data(datas, processer, converter, options)
    write_to_json(options.tgt_path, results)
    


'''
测试集数据生成:
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/test.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NEr \
    --split_num 6 \
    --split test

训练集数据生成:
python ie2instruction/convert_func.py \
    --src_path data/NER/sample.json \
    --tgt_path data/NER/train.json \
    --schema_path data/NER/schema.json \
    --language zh \
    --task NER \
    --split_num 6 \
    --random_sort \
    --split train
'''

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--src_path", type=str, default="data/NER/sample.json")
    parse.add_argument("--tgt_path", type=str, default="data/NER/processed.json")
    parse.add_argument("--schema_path", type=str, default='data/NER/schema.json')
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en'], help="不同语言使用的template及转换脚本不同")
    parse.add_argument("--task", type=str, default="NER", choices=['RE', 'NER', 'EE', 'EET', 'EEA'])
    parse.add_argument("--split", type=str, default='train', choices=['train', 'test'])

    parse.add_argument("--split_num", type=int, default=4, help="单个指令中最大schema数量。默认为4, -1表示不切分, 各个任务推荐的切分数量不同: NER:6, RE:4, EE:4, EET:4, EEA:4")
    parse.add_argument("--neg_schema", type=float, default=1, help="指令中负样本的比例, 默认为1, 即采用全部负样本")
    parse.add_argument("--random_sort", action='store_true', help="是否对指令中的schema随机排序, 默认为False, 即按字母顺序排序")

    options = parse.parse_args()
    process(options)

    