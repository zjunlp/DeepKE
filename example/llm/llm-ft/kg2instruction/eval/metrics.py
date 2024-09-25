from typing import Any
import jieba
from rouge_chinese import Rouge
import rouge
import re
from copy import deepcopy
from collections import defaultdict

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


class Metric:
    def __init__(self, match_mode='normal', language='zh', metrics_list='f1,rouge'):
        self.match_mode = match_mode
        self.language = language
        self.tp = 0.
        self.cnt = 0            
        self.gold_num = 0.        
        self.pred_num = 0.  
        if language == 'zh':
            self.rouge = Rouge()
        self.rouge_1 = 0.   
        self.rouge_2 = 0. 
        self.rouge_l = 0. 
        self.metrics_list = self.init_metrics(metrics_list) 

    def update(self, other_metric):
        self.tp += other_metric.tp
        self.cnt += other_metric.cnt
        self.gold_num += other_metric.gold_num
        self.pred_num += other_metric.pred_num
        self.rouge_1 += other_metric.rouge_1
        self.rouge_2 += other_metric.rouge_2
        self.rouge_l += other_metric.rouge_l

    def init_metrics(self, metrics_list):
        return set(metrics_list.split(","))
    
    def get_rouge_score(self, pred, gold):   
        pred = pred.strip()
        gold = gold.strip()
        if self.language == 'zh':     
            hypothesis = ' '.join(jieba.cut(pred)) 
            reference = ' '.join(jieba.cut(gold))
            if hypothesis == '' and reference != '':
                return 0, 0, 0
            elif hypothesis != '' and reference == '':
                return 0, 0, 0
            elif hypothesis == '' and reference == '':
                return 1, 1, 1
            score = self.rouge.get_scores(hypothesis, reference)
            score = score[0]
        else:
            if pred == '' and gold != '':
                return 0, 0, 0
            elif pred != '' and gold == '':
                return 0, 0, 0
            elif pred == '' and gold == '':
                return 1, 1, 1
            pred = normalize_answer(pred)
            gold = normalize_answer(gold)
            _evaluator = rouge.Rouge()
            try:
                score = _evaluator.get_scores(pred, gold)
            except LookupError:
                raise LookupError
            score = score[0]
        return score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']


    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_score(self):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        f1 = self.safe_div(2 * p * r, p + r)
        score = {'f1': f1 * 100}
        if 'rouge' in self.metrics_list:
            rouge_1 = self.safe_div(self.rouge_1, self.cnt)
            rouge_2 = self.safe_div(self.rouge_2, self.cnt)
            rouge_l = self.safe_div(self.rouge_l, self.cnt)
            score['rouge-1'] = rouge_1
            score['rouge-2'] = rouge_2
            score['rouge-l'] = rouge_l
        return score


    def count_instance(self, gold_text, pred_text,  gold_list, pred_list):
        self.cnt += 1
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

        if 'rouge' in self.metrics_list:
            rouge_1, rouge_2, rouge_l = self.get_rouge_score(pred=pred_text, gold=gold_text)
            self.rouge_1 += rouge_1
            self.rouge_2 += rouge_2
            self.rouge_l += rouge_l


class EEMetric:
    def __init__(self, match_mode='normal', language='zh', metrics_list='f1,rouge'):
        self.match_mode = match_mode
        self.language = language
        self.cnt = 0 
        self.ee_tp = 0.
        self.ee_gold_num = 0.        
        self.ee_pred_num = 0.
        self.ag_tp = 0.
        self.ag_gold_num = 0.
        self.ag_pred_num = 0.
        if language == 'zh':
            self.rouge = Rouge()
        self.rouge_1 = 0.   
        self.rouge_2 = 0. 
        self.rouge_l = 0. 
        self.metrics_list = self.init_metrics(metrics_list) 

    def update(self, other_metric):
        self.cnt += other_metric.cnt
        self.ee_tp += other_metric.ee_tp
        self.ee_gold_num += other_metric.gold_num
        self.ee_pred_num += other_metric.pred_num
        self.ag_tp += other_metric.ag_tp
        self.ag_gold_num += other_metric.ag_gold_num
        self.ag_pred_num += other_metric.ag_pred_num
        self.rouge_1 += other_metric.rouge_1
        self.rouge_2 += other_metric.rouge_2
        self.rouge_l += other_metric.rouge_l

    def init_metrics(self, metrics_list):
        return set(metrics_list.split(","))
    
    def get_rouge_score(self, pred, gold):   
        pred = pred.strip()
        gold = gold.strip()
        if self.language == 'zh':     
            hypothesis = ' '.join(jieba.cut(pred)) 
            reference = ' '.join(jieba.cut(gold))
            if hypothesis == '' and reference != '':
                return 0, 0, 0
            elif hypothesis != '' and reference == '':
                return 0, 0, 0
            elif hypothesis == '' and reference == '':
                return 1, 1, 1
            score = self.rouge.get_scores(hypothesis, reference)
            score = score[0]
        else:
            if pred == '' and gold != '':
                return 0, 0, 0
            elif pred != '' and gold == '':
                return 0, 0, 0
            elif pred == '' and gold == '':
                return 1, 1, 1
            pred = normalize_answer(pred)
            gold = normalize_answer(gold)
            _evaluator = rouge.Rouge()
            try:
                score = _evaluator.get_scores(pred, gold)
            except LookupError:
                raise LookupError
            score = score[0]
        return score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']


    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_score(self):
        ee_p, ee_r = self.safe_div(self.ee_tp, self.ee_pred_num), self.safe_div(self.ee_tp, self.ee_gold_num)
        ee_f1 = self.safe_div(2 * ee_p * ee_r, ee_p + ee_r)
        ag_p, ag_r = self.safe_div(self.ag_tp, self.ag_pred_num), self.safe_div(self.ag_tp, self.ag_gold_num)
        ag_f1 = self.safe_div(2 * ag_p * ag_r, ag_p + ag_r)
        score = {'ee_f1': ee_f1 * 100, 'ag_f1':ag_f1 * 100}
        if 'rouge' in self.metrics_list:
            rouge_1 = self.safe_div(self.rouge_1, self.cnt)
            rouge_2 = self.safe_div(self.rouge_2, self.cnt)
            rouge_l = self.safe_div(self.rouge_l, self.cnt)
            score['rouge-1'] = rouge_1
            score['rouge-2'] = rouge_2
            score['rouge-l'] = rouge_l
        return score


    def count_instance(self, gold_text, pred_text,  gold_list, pred_list):
        self.cnt += 1
        gold_list = [tuple(it) for it in gold_list]
        pred_list = [tuple(it) for it in pred_list]
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
        print(gold_list)
        print(pred_list)

        self.ee_gold_num += len(gold_list)
        self.ee_pred_num += len(pred_list)
        pred_mapper = {}
        gold_mapper = {}
        for it in pred_list:
            pred_mapper[(it[0], it[1])] = set(it[2])
            self.ag_pred_num += len(it[2])
        for it in gold_list:
            gold_mapper[(it[0], it[1])] = set(it[2])
            self.ag_gold_num += len(it[2])


        for pred_ee, pred_ag in pred_mapper.items():
            if pred_ee in gold_mapper:
                self.ee_tp += 1
                for ag in pred_ag:
                    if ag in gold_mapper[pred_ee]:
                        self.ag_tp += 1

        if 'rouge' in self.metrics_list:
            rouge_1, rouge_2, rouge_l = self.get_rouge_score(pred=pred_text, gold=gold_text)
            self.rouge_1 += rouge_1
            self.rouge_2 += rouge_2
            self.rouge_l += rouge_l


