from eval.metric.metric import Metric
from copy import deepcopy


class EEMetric(Metric):
    def __init__(self, match_mode="normal", metrics_list="f1,rouge"):
        super().__init__(match_mode, metrics_list)
        self.ee_tp = 0.
        self.ee_gold_num = 0.        
        self.ee_pred_num = 0.
        self.ag_tp = 0.
        self.ag_gold_num = 0.
        self.ag_pred_num = 0.

    def compute_f1(self):
        result = {}
        result['总样本数'] = self.f1_cnt
        result['错误数'] = self.error
        result['event_P'] = self.safe_div(self.ee_tp, self.ee_pred_num)
        result['event_R'] = self.safe_div(self.ee_tp, self.ee_gold_num)
        result['event_F1'] = self.safe_div_(2 *  result['event_P'] * result['event_R'],  result['event_P'] + result['event_R'])
        
        result['argument_P'] = self.safe_div(self.ag_tp, self.ag_pred_num)
        result['argument_R'] = self.safe_div(self.ag_tp, self.ag_gold_num)
        result['argument_F1'] = self.safe_div_(2 * result['argument_P'] * result['argument_R'], result['argument_P'] + result['argument_R'])
        return result


    def count_instance_evt(self, gold_list, pred_list):
        gold_list = [tuple(it) for it in gold_list]
        pred_list = [tuple(it) for it in pred_list]
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
        dup_gold_list = deepcopy(gold_list)
        dup_pred_list = deepcopy(pred_list)

        self.ee_pred_num += len(pred_list)
        self.ee_gold_num += len(gold_list)

        for pred in pred_list:
            if pred in dup_gold_list:
                self.ee_tp += 1
                dup_gold_list.remove(pred)
                dup_pred_list.remove(pred)


    def count_instance_arg(self, gold_list, pred_list):
        gold_list = [tuple(it) for it in gold_list]
        pred_list = [tuple(it) for it in pred_list]
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
        dup_gold_list = deepcopy(gold_list)
        dup_pred_list = deepcopy(pred_list)

        self.ag_pred_num += len(pred_list)
        self.ag_gold_num += len(gold_list)

        for pred in pred_list:
            if pred in dup_gold_list:
                self.ag_tp += 1
                dup_gold_list.remove(pred)
                dup_pred_list.remove(pred)


    def count_instance_f1(self, gold_list, pred_list):
        self.f1_cnt += 1
        gold_evt = []
        pred_evt = []
        gold_args = []
        pred_args = []

        for evt_type, trigger, args in gold_list:
            gold_evt.append((evt_type, trigger))
            for name, stype in args: 
                gold_args.append((evt_type, name, stype))

        for evt_type, trigger, args in pred_list:
            pred_evt.append((evt_type, trigger))
            for name, stype in args: 
                pred_args.append((evt_type, name, stype))

        self.count_instance_evt(gold_evt, pred_evt)
        self.count_instance_arg(gold_args, pred_args)
        
    