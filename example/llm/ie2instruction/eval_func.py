# coding=utf-8
import json
import os
import sys
sys.path.append('./')
import argparse
from collections import defaultdict
from eval.metric import get_metric
from eval.extracter import get_extracter
from convert.utils.constant import NER, RE, EE, EEA, EET, KG, SPO


def convert_kg(outputs, task):
    kgs = []
    if task == NER:
        for it in outputs:
            kgs.append((it['entity'], it['entity_type']))
    elif task == RE or task == KG:
        for it in outputs:
            kgs.append((it.get('head', ''), it.get('relation', ''), it.get('tail', '')))
    elif task == SPO:
        for it in outputs:
            kgs.append((it.get('head_type', ''), it.get('head', ''), it.get('relation', ''), it.get('tail_type', ''), it.get('object', '')))
    elif task == EE:
        for it in outputs:
            args = []
            for arg in it['arguments']:
                args.append((arg['argument'], arg['role']))
            kgs.append((it['event_type'], it['event_trigger'], tuple(args)))
    elif task == EEA:
        for it in outputs:
            args = []
            for arg in it['arguments']:
                args.append((arg['argument'], arg['role']))
            kgs.append((it['event_type'], '', tuple(args)))
    elif task == EET:
        for it in outputs:
            kgs.append((it['event_trigger'], it['event_type']))
    return kgs


def evaluate(options):
    extracter_class = get_extracter(options.task)
    metric_class = get_metric(options.task)
    
    mapper = defaultdict(dict)
    with open(options.path1, 'r') as reader:
        for line in reader:
            data = json.loads(line)
            iid = data.get('source', 'None') + data['id']
            if iid in mapper:
                mapper[iid]['output'].append(data['output'])
            else:
                instr = json.loads(data['instruction'])
                inpt = instr['input']
                mapper[iid] = {'output':[data['output'], ], 'label':json.loads(data['label']), 'source':data.get('source', 'None'), 'input':inpt}


    if options.sort_by != "":
        cate_set = set()
        for key, record in mapper.items():
            cate_set.add(record[options.sort_by])
        cate_dict = {}
        for cate in cate_set:
            cate_dict[cate] = metric_class(options.match_mode, options.metrics_list)
    total_counter = metric_class(options.match_mode, options.metrics_list)


    extracter = extracter_class()
    for key, value in mapper.items():
        preds = value['output']
        label = value['label']

        converted_preds = []
        for it in preds:
            flag, out_rst = extracter.extract(it)
            if not flag:
                if options.sort_by:
                    cate_dict[value[options.sort_by]].count_error()
                total_counter.count_error()
            converted_preds.extend(out_rst)
        label_kgs = convert_kg(label, options.task)

        if options.sort_by:
            cate_dict[value[options.sort_by]].count_instance(
                gold_list=label_kgs, pred_list=converted_preds
            )
        total_counter.count_instance(
            gold_list=label_kgs, pred_list=converted_preds
        )


    # 写入结果到json文件
    cate_results = {}
    if options.sort_by:
        cate_dict = dict(sorted(cate_dict.items()))
        for key, cate_counter in cate_dict.items():
            cate_results[key] = cate_counter.compute()
    total_result = total_counter.compute()

    all_result = {}
    all_result['total'] = total_result
    for key, value in cate_results.items():
        all_result[key] = value
    print(all_result)
    json.dump(all_result, open(options.out_path, 'w'), ensure_ascii=False, indent=4)



def main():
    '''
    python ie2instruction/eval_func.py \
        --path1 results/baichuan2-13b-iepile-lora_output.json \
        --task RE \
        --sort_by source 
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument("--path1", type=str, default="")
    parse.add_argument("--task", type=str, default='re', choices=['NER', 'RE', 'EE', 'SPO', 'EET', 'EEA', 'KG', 'MRC'])
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en']) 
    parse.add_argument("--NAN", type=str, default="")
    parse.add_argument("--prefix", type=str, default='')
    parse.add_argument("--sort_by", type=str, default='')
    parse.add_argument("--match_mode", type=str, default="normal")
    parse.add_argument("--metrics_list", type=str, default="f1")
    options = parse.parse_args()

    dname = os.path.dirname(options.path1)
    fname = os.path.basename(options.path1)
    os.makedirs(os.path.join(dname, "output_msg"), exist_ok=True)
    options.out_path = os.path.join(dname, "output_msg", fname)
    evaluate(options)



if __name__=="__main__":
    main()
