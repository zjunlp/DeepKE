from rouge_chinese import Rouge
import re
import jieba
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


class F1Metric:
    def __init__(self, match_mode="normal"):
        self.match_mode = match_mode
        self.f1_cnt = 0          
        self.error = 0 

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return round(a / b * 100, 2)

    @staticmethod
    def safe_div_(a, b):
        if b == 0.:
            return 0.
        else:
            return round(a / b, 2)

    def count_error(self):
        self.error += 1

    
    def count_instance_f1(self, gold_list, pred_list):
        raise NotImplementedError
    
    def compute_f1(self):
        raise NotImplementedError



class RougeMetric:
    def __init__(self): 
        self.rouge_cnt = 0 
        self.rouge = Rouge()
        self.rouge_1 = 0.   
        self.rouge_2 = 0. 
        self.rouge_l = 0. 


    def get_rouge_score(self, pred_text, gold_text):   
        pred_text = pred_text.strip()
        gold_text = gold_text.strip()    
        hypothesis = ' '.join(jieba.cut(pred_text)) 
        reference = ' '.join(jieba.cut(gold_text))
        if hypothesis == '' and reference != '':
            return 0, 0, 0
        elif hypothesis != '' and reference == '':
            return 0, 0, 0
        elif hypothesis == '' and reference == '':
            return 1, 1, 1
        score = self.rouge.get_scores(hypothesis, reference)
        score = score[0]
        return score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']


    def count_instance_rouge(self, gold_text, pred_text):
        self.rouge_cnt += 1
        rouge_1, rouge_2, rouge_l = self.get_rouge_score(pred=pred_text, gold=gold_text)
        self.rouge_1 += rouge_1
        self.rouge_2 += rouge_2
        self.rouge_l += rouge_l


    def compute_rouge(self):
        score = {}
        rouge_1 = self.safe_div(self.rouge_1, self.rouge_cnt)
        rouge_2 = self.safe_div(self.rouge_2, self.rouge_cnt)
        rouge_l = self.safe_div(self.rouge_l, self.rouge_cnt)
        score['rouge-1'] = rouge_1
        score['rouge-2'] = rouge_2
        score['rouge-l'] = rouge_l
        return score


    

class Metric(F1Metric, RougeMetric):
    def __init__(self, match_mode='normal', metrics_list='f1,rouge'):
        F1Metric.__init__(self, match_mode=match_mode)  
        RougeMetric.__init__(self) 
        self.metrics_list = self.init_metrics(metrics_list) 

    def init_metrics(self, metrics_list):
        return set(metrics_list.split(","))
    

    def compute(self):
        score = {}
        if 'f1' in self.metrics_list:
            f1_socre = self.compute_f1()
            score.update(f1_socre)
        if 'rouge' in self.metrics_list:
            rouge_score = self.compute_rouge()
            score.update(rouge_score)
        return score


    def count_instance(self, gold_list, pred_list, gold_text="", pred_text=""):
        if 'f1' in self.metrics_list:
            self.count_instance_f1(gold_list, pred_list)
        if 'rouge' in self.metrics_list:
            self.count_instance_rouge(gold_text, pred_text)
    
    