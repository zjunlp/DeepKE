import yaml
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
        if options.cluster_mode:
            total_schemas = processer.negative_cluster_sample(record, options.split_num, options.random_sort)
        else:
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
    if options.cluster_mode:
        processer.set_hard_dict(json.load(open(options.hard_negative_path, 'r')))
    processer.set_negative(options.neg_schema)

    options.source = options.src_path.split('/')[-2]  # 用源路径的最后一个文件夹名作为source
    datas = processer.read_data(options.src_path)
    if options.split == 'test':
        results = get_test_data(datas, processer, options)
    else:
        results = get_train_data(datas, processer, converter, options)
    write_to_json(options.tgt_path, results)


def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default=None)
    parser.add_argument("--tgt_path", type=str, default=None)
    parser.add_argument("--schema_path", type=str, default=None)
    parser.add_argument("--hard_negative_path", type=str, default=None)
    parser.add_argument("--cluster_mode", action='store_true', help="是否使用cluster模式")
    parser.add_argument("--language", type=str, default=None, choices=['zh', 'en'])
    parser.add_argument("--task", type=str, default=None, choices=['RE', 'NER', 'EE', 'EET', 'EEA', 'SPO', 'KG'])
    parser.add_argument("--split", type=str, default=None, choices=['train', 'test'])
    parser.add_argument("--split_num", type=int, default=4)
    parser.add_argument("--neg_schema", type=float, default=1.0)
    parser.add_argument("--random_sort", action='store_true', help="是否对指令中的schema随机排序")
    args = parser.parse_args()

    # if user don't pass any parameters, try to load configuration from yaml
    if not any([args.src_path, args.tgt_path, args.schema_path, args.language, args.task]):
        yaml_config = load_config_from_yaml('examples/infer/convert.yaml')
        for key, value in yaml_config.items():
            if value is not None:
                setattr(args, key, value)

    return args


if __name__ == "__main__":
    options = parse_args()
    process(options)
