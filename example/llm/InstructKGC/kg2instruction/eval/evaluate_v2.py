# coding=utf-8
import json
import os
import sys
import re
sys.path.append('./')
import argparse


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
def normalize_answer(s):
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s

def remove_yinhao(s):
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    return s.lower()


def low(s):
    if type(s) == str:
        return normalize_answer(s)
    else:
        for it in s:
            it = low(it)
        return s


def postprocess(kgs, language="zh"):
    if kgs == None:
        return None
    new_kgs = []
    for kg in kgs:
        head, relation, tail = kg
        head = remove_yinhao(head)
        relation = remove_yinhao(relation)
        tail = remove_yinhao(tail)
        if head == "" or relation == "" or tail == "":
            continue
        new_kgs.append((head, relation, tail))
    if language == "en":
        new_kgs = low(new_kgs)
    return new_kgs


def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file, ensure_ascii=False, indent=4)


def kg_convert(kgs, task):
    if task == 'RE':
        new_kags = []
        for it in kgs:
            if len(it) != 3:
                continue
            new_kags.append({"head":it[0], "relation":it[1], "tail":it[2]})
    elif task == 'NER':
        new_kags = []
        for it in kgs:
            if len(it) != 2:
                continue
            new_kags.append({"entity":it[0], "entity_type":it[1]})
    elif task == 'EE' or task == 'EEA':
        new_kags = []
        for it in kgs:
            if len(it) != 3:
                continue
            new_args = []
            for iit in it[2]:
                new_args.append({"argument":iit[0], "role":iit[1]})
            new_kags.append({"event_trigger":it[0], "event_type":it[1], "arguments":new_args})
    elif task == 'EET':
        new_kags = []
        for it in kgs:
            if len(it) != 2:
                continue
            new_kags.append({"event_type":it[0], "event_trigger":it[1]})
    else:
        raise KeyError
    return new_kags


def evaluate_file(standard_path, submit_path, extractor, converter, MyMetric, options):
    error = []
    standard_mapper = {}
    with open(standard_path, "r") as f1:
        for line in f1:
            data = json.loads(line.strip())
            kg = extractor.extract(data['input'], data['output'])
            if kg is None:
                error.append(("pred", data["output"]))
                kg = []
                #stand_output = data["output"]
            else:
                if options.task == "RE":
                    kg = postprocess(kg, options.language)
                #new_kgs = kg_convert(kg, options.task)
                #_, stand_output = converter.convert(new_kgs, rand1=0, rand2=0)
            kg = low(kg)
            data['kg'] = kg
            #data['output'] = stand_output.lower().strip()
            standard_mapper[data["input"]] = data


    submit_mapper = {}
    with open(submit_path, "r") as f1:
        for i, line in enumerate(f1):
            try:
                data = json.loads(line.strip())
            except json.decoder.JSONDecodeError:
                print(i)
            kg = extractor.extract(data['input'], data['output'])
            if kg is None:
                error.append(("pred", data["output"]))
                kg = []
                #stand_output = data["output"]
            else:
                if options.task == "RE":
                    kg = postprocess(kg, options.language)
                #new_kgs = kg_convert(kg, options.task)
                #_, stand_output = converter.convert(new_kgs, rand1=0, rand2=0)
            kg = low(kg)
            data['kg'] = kg
            #data['output'] = stand_output.lower().strip()
            submit_mapper[data["input"]] = data

    cate_dict = {}
    total_metric = MyMetric(options.match_mode, options.language, metrics_list=options.metrics_list)
    if options.sort_by != "":
        cate_set = set()
        for key, gold_record in standard_mapper.items():
            cate_set.add(gold_record[options.sort_by])
        for cate in cate_set:
            cate_dict[cate] = MyMetric(options.match_mode, options.language, metrics_list=options.metrics_list)
    


    miss_list = []
    for key, gold_record in standard_mapper.items():
        try:
            pred_record = submit_mapper[key]
        except KeyError:
            miss_list.append(key)
        else:
            #pred_record["kg"] = rule_process(listtuple2listlist(pred_record["kg"]), gold_record['input'], gold_record['cate'])
            if options.sort_by != "":
                cate_dict[gold_record[options.sort_by]].count_instance(
                    gold_list=gold_record["kg"], 
                    pred_list=pred_record["kg"],
                    gold_text=gold_record["output"],
                    pred_text=pred_record["output"],
                )
            total_metric.count_instance(
                gold_list=gold_record["kg"], 
                pred_list=pred_record["kg"],
                gold_text=gold_record["output"],
                pred_text=pred_record["output"],
            )

    diff_result = diff(options, standard_mapper, submit_mapper)
    return total_metric, cate_dict, miss_list, error, diff_result



def evaluate(options):
    MyMetric = Metric
    if options.task == "NER":
        extractor = NERExtractor(options.language, options.NAN, options.prefix)
        converter = NERConverter(options.language, options.NAN, options.prefix)
    elif options.task == "RE":
        extractor = REExtractor(options.language, options.NAN, options.prefix)
        converter = REConverter(options.language, options.NAN, options.prefix)
    elif options.task == "EE":
        extractor = EEExtractor(options.language, options.NAN, options.prefix)
        converter = EEConverter(options.language, options.NAN, options.prefix)
        MyMetric = EEMetric
    elif options.task == "EET":
        extractor = EETExtractor(options.language, options.NAN, options.prefix)
        converter = EETConverter(options.language, options.NAN, options.prefix)
    elif options.task == "EEA":
        extractor = EEAExtractor(options.language, options.NAN, options.prefix)
        converter = EEAConverter(options.language, options.NAN, options.prefix)
        MyMetric = EEMetric
    else:
        raise KeyError
    
    total_metric = MyMetric(options.match_mode, options.language, metrics_list=options.metrics_list)
    cate_dict = {}
    miss_list = []
    error = []
    if os.path.isfile(options.standard_path):
        total_metric, cate_dict, miss_list, error, diff_result = evaluate_file(options.standard_path, options.submit_path, extractor, converter, MyMetric, options)
    else:
        for fl in os.listdir(options.standard_path):
            if fl.endswith(".json"):
                standard_path = os.path.join(options.standard_path, fl)
                submit_path = os.path.join(options.submit_path, fl)
                total_metric_, cate_dict_, miss_list_, error_, diff_result_ = evaluate_file(standard_path, submit_path, extractor, converter, options)
                total_metric.update(total_metric_)
                diff_result.append({fl : diff_result_})
                for key, value in cate_dict_.items():
                    if key not in cate_dict:
                        cate_dict[key] = value
                    else:
                        cate_dict[key].update(value)
                miss_list.extend(miss_list_)
                error.extend(error_)

    all_result = {}
    total_score = total_metric.compute_score()
    print("total_score: ", total_score)
    all_result["total_score"] = total_score
    all_result["diff_result"] = diff_result
    all_result["error"] = len(error)
    for key, value in cate_dict.items():
        cate_score = value.compute_score()
        all_result[key] = cate_score
        print(key, cate_score)
    json.dump(all_result, open(options.out_path, 'w'), ensure_ascii=False, indent=4)

    print("eeror: ", error)
    print("eeror Number: ", len(error))
    if len(miss_list) != 0:
        print("Miss:", len(miss_list))



def main():
    '''
    python eval/evaluate_v2.py \
        --standard_path /newdisk3/data/guihh/alpaca/alpaca-lora/data/test/zh/valid1_zh_sample_schema_v3.json \
        --submit_path /newdisk3/data/guihh/alpaca/alpaca-lora/results/ccks_new/baichuan/Baichuan2-7B-Base-4bit-filter-all-v3-zh-new-valid1_zh_sample_schema_v3.json \
        --task RE \
        --NAN 'NAN' \
        --language zh \
        --metrics_list f1,rouge \
        --sort_by cate \
        --sort_by_relation 
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument("--standard_path", type=str, default="/nature/ghh/alpaca/alpaca-lora/data/multi/valid_all_1.json")
    parse.add_argument("--submit_path", type=str, default="/nature/ghh/alpaca/alpaca-lora/results/ccks/llama/llama2-7b-4bit-wiki-kuohao-valid_all_kuohao.json")
    parse.add_argument("--task", type=str, default='RE', choices=['NER', 'RE', 'EE', 'EET', 'EEA'])
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en']) 
    parse.add_argument("--NAN", type=str, default="")
    parse.add_argument("--prefix", type=str, default='')
    parse.add_argument("--sort_by", type=str, default='')
    parse.add_argument("--sort_by_relation", action="store_true", default=False)
    parse.add_argument("--match_mode", type=str, default="set")
    parse.add_argument("--metrics_list", type=str, default="f1")
    options = parse.parse_args()

    dname = os.path.dirname(options.submit_path)
    fname = os.path.basename(options.submit_path)
    os.makedirs(os.path.join(dname, "output_msg"), exist_ok=True)
    os.makedirs(os.path.join(dname, "diff"), exist_ok=True)
    options.out_path = os.path.join(dname, "output_msg", fname)
    options.diff_path = os.path.join(dname, "diff", fname)

    evaluate(options)
    #diff(options, standard_mapper, submit_mapper)



if __name__=="__main__":
    main()
