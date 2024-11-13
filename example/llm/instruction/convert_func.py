import sys
sys.path.append("./")
import yaml
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
        total_schemas = multischema_split_by_num_test(schemas, options['split_num'])
        for schema in total_schemas:
            sinstruct = multischema_construct_instruction(options['task'], options['language'], schema, record['text'])
            record2 = {
                'id': iid,
                'task': options['task'],
                'source': options['source'],
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
        if options.get('cluster_mode'):
            total_schemas = processer.negative_cluster_sample(record, options['split_num'], options.get('random_sort', False))
        else:
            total_schemas = processer.negative_sample(record, options['split_num'], options.get('random_sort', False))
        task_record = processer.get_task_record(record)
        output_texts = convert_output(converter, record['text'], total_schemas, task_record)    # Split `schema` and `output_text` according to `split_num`
        for schema, output_text in zip(total_schemas, output_texts):
            sinstruct = multischema_construct_instruction(options['task'], options['language'], schema, record['text'])
            record2 = {
                'task': options['task'],
                'source': options['source'],
                'instruction': sinstruct,
                'output': output_text
            }
            results.append(record2)
    return results


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def process(options):
    converter = get_converter(options['task'])(options['language'], NAN='NAN')
    processer_class = get_processer(options['task'])
    processer = processer_class.read_from_file(
        processer_class, options['schema_path'], negative=-1
    )
    if options.get('cluster_mode'):
        processer.set_hard_dict(json.load(open(options['hard_negative_path'], 'r')))
    processer.set_negative(options.get('neg_schema', -1))

    options['source'] = options['src_path'].split('/')[-2]  # Use the last folder name in the source path as `source`
    datas = processer.read_data(options['src_path'])
    if options['split'] == 'test':
        results = get_test_data(datas, processer, options)
    else:
        results = get_train_data(datas, processer, converter, options)
    write_to_json(options['tgt_path'], results)


if __name__ == "__main__":
    config = load_config('examples/fine_turning/convert.yaml')
    mode = config.get('mode', 'train')
    options = config[mode]

    process(options)

