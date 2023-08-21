from copy import deepcopy


class Metric:
    def __init__(self, match_mode='normal'):
        self.match_mode = match_mode
        self.tp = 0.
        self.cnt = 0            
        self.gold_num = 0.        
        self.pred_num = 0.    
        self.rougen_2 = 0.  

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
        score = {'f1': f1}
        return score

    def count_instance_f1(self, gold_list, pred_list):
        gold_list = [tuple(it) for it in gold_list]
        pred_list = [tuple(it) for it in pred_list]
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)


