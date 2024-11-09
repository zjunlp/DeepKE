from eval.metric.metric import Metric
from copy import deepcopy


class NERMetric(Metric):
    def __init__(self, match_mode='normal', metrics_list='f1,rouge'):
        super().__init__(match_mode, metrics_list)
        self.pred_num = 0
        self.gold_num = 0
        self.tp = 0    



    def compute_f1(self):
        result = {}
        result['总样本数'] = self.f1_cnt
        result['错误数'] = self.error
        result['P'] = self.safe_div(self.tp, self.pred_num)
        result['R'] = self.safe_div(self.tp, self.gold_num)
        result['F1'] = self.safe_div_(2 * result['P'] * result['R'], result['P'] + result['R'])
        return result
    

    def count_instance_f1(self, gold_list, pred_list):
        self.f1_cnt += 1
        gold_list = [tuple(it) for it in gold_list]
        pred_list = [tuple(it) for it in pred_list]
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
        dup_gold_list = deepcopy(gold_list)
        dup_pred_list = deepcopy(pred_list)
        
        self.pred_num += len(pred_list)
        self.gold_num += len(gold_list)

        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)
                dup_pred_list.remove(pred)

