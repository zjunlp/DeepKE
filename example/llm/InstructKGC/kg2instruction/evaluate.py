# coding=utf-8
import sys
sys.path.append('./')
import json
import re
import argparse

from eval.metrics import Metric, EEMetric
from eval.extracter import NERExtractor, REExtractor, EEExtractor, EETExtractor, EEAExtractor


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
def normalize_answer(s):
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s

def low(s):
    if type(s) == str:
        return normalize_answer(s)
    else:
        for it in s:
            it = low(it)
        return s
    

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file, ensure_ascii=False, indent=4)


def evaluate(options):
    MyMetric = Metric
    if options.task == "NER":
        extractor = NERExtractor(options.language, options.NAN, options.prefix)
    elif options.task == "RE":
        extractor = REExtractor(options.language, options.NAN, options.prefix)
    elif options.task == "EE":
        extractor = EEExtractor(options.language, options.NAN, options.prefix)
        MyMetric = EEMetric
    elif options.task == "EET":
        extractor = EETExtractor(options.language, options.NAN, options.prefix)
    elif options.task == "EEA":
        extractor = EEAExtractor(options.language, options.NAN, options.prefix)
        MyMetric = EEMetric
    else:
        raise KeyError
    

    error = []
    standard_mapper = {}
    with open(options.standard_path, "r") as f1:
        for line in f1:
            data = json.loads(line.strip())
            kg = extractor.extract(data['input'], data['output'])
            if kg is None:
                error.append(("gold", data["output"]))
                kg = []
            kg = low(kg)
            data['kg'] = kg
            standard_mapper[data["input"]] = data

    submit_mapper = {}
    with open(options.submit_path, "r") as f1:
        for line in f1:
            data = json.loads(line.strip())
            kg = extractor.extract(data['input'], data['output'])
            if kg is None:
                error.append(("pred", data["output"]))
                kg = []
            kg = low(kg)
            data['kg'] = kg
            submit_mapper[data["input"]] = data
    

    metric = MyMetric(options.match_mode, options.language, metrics_list=options.metrics_list)
    miss_list = []
    for key, gold_record in standard_mapper.items():
        try:
            pred_record = submit_mapper[key]
        except KeyError:
            miss_list.append(key)
        else:
            metric.count_instance(
                gold_list=gold_record["kg"], 
                pred_list=pred_record["kg"],
                gold_text=gold_record["output"],
                pred_text=pred_record["output"],
            )

    score = metric.compute_score()
    print(score)

    print("eeror", error)
    if len(miss_list) != 0:
        print("Miss:", len(miss_list))



def main():
    '''
    python kg2instruction/evaluate.py \
        --standard_path data/NER/processed.json \
        --submit_path data/NER/processed.json \
        --task NER \
        --language zh
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument("--standard_path", type=str, default="data/NER/processed.json")
    parse.add_argument("--submit_path", type=str, default="results/ner_test.json")
    parse.add_argument("--task", type=str, default='RE', choices=['NER', 'RE', 'EE', 'EET', 'EEA'])
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en'])
    parse.add_argument("--NAN", type=str, default="")
    parse.add_argument("--prefix", type=str, default='')
    parse.add_argument("--match_mode", type=str, default="set")
    parse.add_argument("--metrics_list", type=str, default="f1")
    options = parse.parse_args()

    evaluate(options)


if __name__=="__main__":
    main()
