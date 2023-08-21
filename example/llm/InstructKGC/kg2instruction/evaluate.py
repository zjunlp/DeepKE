# coding=utf-8
import sys
sys.path.append('./')
import json
import os
import re
import argparse

from re_template import re_post_process_en, re_post_process_zh
from ner_template import ner_post_process_en, ner_post_process_zh
from ee_template import ee_post_process_en, ee_post_process_zh
from metrics import Metric


def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file, ensure_ascii=False, indent=4)



def evaluate(options):
    error = []
    standard_mapper = {}
    if options.language == 'zh':
        ner_post_process = ner_post_process_zh
        re_post_process = re_post_process_zh
        ee_post_process = ee_post_process_zh
    else:
        ner_post_process = ner_post_process_en
        re_post_process = re_post_process_en
        ee_post_process = ee_post_process_en
    with open(options.standard_path, "r") as f1:
        for line in f1:
            data = json.loads(line.strip())
            if options.task == 'ner':
                kg = ner_post_process(data['output'])
            elif options.task == 're':
                kg = re_post_process(data['output'])
            elif options.task == 'ee':
                kg = ee_post_process(data['output'])
            else:
                raise KeyError
            if kg is None:
                error.append(("gold", data["output"]))
                kg = []
            data['kg'] = kg
            standard_mapper[data["input"]] = data

    submit_mapper = {}
    with open(options.submit_path, "r") as f1:
        for line in f1:
            data = json.loads(line.strip())
            if options.task == 'ner':
                kg = ner_post_process(data['output'])
            elif options.task == 're':
                kg = re_post_process(data['output'])
            elif options.task == 'ee':
                kg = ee_post_process(data['output'])
            else:
                raise KeyError
            if kg is None:
                error.append(("pred", data["output"]))
                kg = []
            data['kg'] = kg
            submit_mapper[data["input"]] = data
    

    metric = Metric(options.match_mode)
    miss_list = []
    for key, gold_record in standard_mapper.items():
        try:
            pred_record = submit_mapper[key]
        except KeyError:
            miss_list.append(key)
        else:
            metric.count_instance_f1(gold_list=gold_record["kg"], pred_list=pred_record["kg"])

    score = metric.compute_f1()
    print(key, score)

    print("eeror", error)
    if len(miss_list) != 0:
        print("Miss:", len(miss_list))
    return standard_mapper, submit_mapper




def main():
    '''
    python kg2instruction/evaluate.py \
        --standard_path data/NER/processed.json \
        --submit_path data/NER/processed.json \
        --task ner \
        --language zh
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument("--standard_path", type=str, default="data/NER/processed.json")
    parse.add_argument("--submit_path", type=str, default="results/ner_test.json")
    parse.add_argument("--task", type=str, default='ee', choices=['ner', 're', 'ee'])
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en'])
    parse.add_argument("--match_mode", type=str, default="normal")
    options = parse.parse_args()

    standard_mapper, submit_mapper = evaluate(options)


if __name__=="__main__":
    main()
