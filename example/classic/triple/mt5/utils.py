from datasets import Features, Value, Sequence
import re
from typing import List
from copy import deepcopy


RecordFeature = Features({
    'input_ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
    'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
    'labels': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
})

RE_triple = re.compile('(\([^\(\)]*\)?)')


class Metric:
    def __init__(self):
        self.tp = 0.        
        self.gold_num = 0.        
        self.pred_num = 0.    

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b
    
    def compute_f1(self):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        f1 = self.safe_div(2 * p * r, p + r)
        return f1 * 100


    def count_instance_f1(self, gold_list, pred_list):
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        if len(gold_list) > 0 and len(pred_list) > 0:
            assert len(gold_list[0]) == len(pred_list[0])

        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)


def rte_post_process(result):    
    '''对返回结果进行后处理, 清除\n等多余字符; 匹配(,,)格式; 获得头、尾实体和关系'''
    def clean(spt, i):
        try:
            s = spt[i]
        except IndexError:
            s = ""
        return s
    rst = result.replace("\n", "")
    matches = re.findall(RE_triple, rst)
    new_record = []
    for m in matches:
        spt = m.split(",")
        new_record.append([clean(spt, 0), clean(spt, 1), clean(spt, 2)])
    return new_record


def get_extract_metrics_f1(golds_outtext : List, preds_outtext : List):
    relation_metric = Metric()

    for gold_outtext, pred_outtext in zip(golds_outtext, preds_outtext):
        gold_kg = rte_post_process(gold_outtext)
        pred_kg = rte_post_process(pred_outtext)
        relation_metric.count_instance_f1(gold_kg, pred_kg)

    f1 = relation_metric.compute_f1()
    result = {'overall-score':  f1}
    return result
