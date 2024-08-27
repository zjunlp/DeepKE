"""
    Evaluation metrics for structured prediction.
"""
import logging
import numpy as np
from collections import Counter, defaultdict
from scipy.optimize import linear_sum_assignment

from .blanc import blanc, tuple_to_metric

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class PRFEvaluator(object):
    def __init__(self):
        self.tp = 1e-6
        self.fn = 1e-6
        self.fp = 1e-6

    def update(self, predicted, gold):
        predicted = " ".join(predicted['coref']) if predicted['coref'] else "PRED_NONE"
        
        gold_coref = " ".join(gold['coref']) if gold['coref'] else "GOLD_NONE"
        gold_a, gold_b = " ".join(gold['A-coref']), " ".join(gold['B-coref'])        
        try:
            if predicted in gold_coref or gold_coref in predicted:
                self.tp += 1
            elif (predicted in gold_a) or (gold_a in predicted):
                self.fp += 1
            elif (predicted in gold_b) or (gold_b in predicted):
                self.fp += 1
            elif gold_coref!="GOLD_NONE":
                self.fn += 1
        except:
            pass
        return
    
    def get_p(self, ):
        return self.tp / (self.fp + self.tp)
    
    def get_r(self, ):
        return self.tp / (self.fn + self.tp)

    def get_f1(self, ):
        return (2*self.get_p()*self.get_r()) / (self.get_p() + self.get_r())


class NEREvaluator(object):
    def __init__(self):
        self.tp = 1e-6
        self.fn = 1e-6
        self.fp = 1e-6

    def update(self, predicted, gold):
        self.tp += len(set(predicted) & set(gold))
        self.fn += len(set(gold) - set(predicted))
        self.fp += len(set(predicted) - set(gold))
        
        return
    
    def get_p(self, ):
        return self.tp / (self.fp + self.tp)
    
    def get_r(self, ):
        return self.tp / (self.fn + self.tp)

    def get_f1(self, ):
        return (2*self.get_p()*self.get_r()) / (self.get_p() + self.get_r())


class MentionEvaluator(object):
    def __init__(self):
        self.total = 1e-6
        self.recalled = 1e-6
        self.predicted = 1e-6
        
        self.word_num = 1e-6

    def update(self, predicted, gold, word_num):
        self.word_num += word_num
        
        self.total += len(gold)
        self.predicted += len(predicted)
        self.recalled += len(set(gold) & set(predicted))
        return
    
    def get_mention_recall(self, ):
        logger.info("total: %f, recalled: %f, predicted: %f, ratio: %f"%(self.total, self.recalled, self.predicted, self.predicted/self.word_num))
        return self.recalled / self.total


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]
        
        self.c_tuple = [0,0,0]
        self.n_tuple = [0,0,0]

    def update(
        self, predicted, gold, mention_to_predicted, mention_to_gold, **kwargs
    ):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)
        
        c_tuple, n_tuple = blanc(gold, predicted)
        for i in range(3):
            self.c_tuple[i] += c_tuple[i]
        for i in range(3):
            self.n_tuple[i] += n_tuple[i]

    def get_all(self):
        all_res = {}
        name_dict = {0: "muc", 1: "b_cubed", 2: "ceafe"}
        for i, e in enumerate(self.evaluators):
            all_res[name_dict[i]+"_f1"] = e.get_f1()
            all_res[name_dict[i]+"_p"] = e.get_precision()
            all_res[name_dict[i]+"_r"] = e.get_recall()
            
        return all_res
    
    def get_f1(self):
        for e in self.evaluators:
            print("f:", e.get_f1())
        
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        for e in self.evaluators:
            print("r:", e.get_recall())
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        for e in self.evaluators:
            print("p:", e.get_precision())
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    
    def get_blanc_prf(self):
        blanc_scores = tuple_to_metric(self.c_tuple, self.n_tuple)
        blanc_p, blanc_r, blanc_f = tuple(0.5*(a+b) for (a,b) in zip(*blanc_scores))
        return blanc_p, blanc_r, blanc_f
    
    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1(), self.get_blanc_prf()


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(
        self, 
        predicted, gold, mention_to_predicted, mention_to_gold,
        **kwargs
    ):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
           continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            # if len(c2) != 1:
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_sum_assignment(-scores)
    matching = np.transpose(np.asarray(matching))
    similarity = sum(scores[matching[:,0], matching[:,1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem



class EREEvaluator(object):
    def __init__(self):
        self.ent_tp = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6 #
        self.ent_fn = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6
        self.ent_fp = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6
        
        self.rel_tp = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6
        self.rel_fn = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6
        self.rel_fp = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6

        self.rel_p_tp = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6
        self.rel_p_fn = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6
        self.rel_p_fp = 1e-6 # defaultdict(lambda: 1e-6)# 1e-6

    def update(
        self, 
        predicted_entities, predicted_relations,
        gold_entities, gold_relations
    ):
        self.ent_tp += len(set(predicted_entities) & set(gold_entities))
        self.ent_fn += len(set(gold_entities) - set(predicted_entities))
        self.ent_fp += len(set(predicted_entities) - set(gold_entities))

        self.rel_p_tp += len(set(predicted_relations) & set(gold_relations))
        self.rel_p_fn += len(set(gold_relations) - set(predicted_relations))
        self.rel_p_fp += len(set(predicted_relations) - set(gold_relations))

        reduced_predicted_relations = [x[:3] for x in predicted_relations]
        reduced_gold_relations = [x[:3] for x in gold_relations]

        self.rel_tp += len(set(reduced_predicted_relations) & set(reduced_gold_relations))
        self.rel_fn += len(set(reduced_gold_relations) - set(reduced_predicted_relations))
        self.rel_fp += len(set(reduced_predicted_relations) - set(reduced_gold_relations))

        return
    
    def get_p(self, average='micro'):
        return (
            self.ent_tp / (self.ent_fp + self.ent_tp), 
            self.rel_tp / (self.rel_fp + self.rel_tp), 
            self.rel_p_tp / (self.rel_p_fp + self.rel_p_tp)
        )
    
    def get_r(self, average='micro'):
        return (
            self.ent_tp / (self.ent_fn + self.ent_tp), 
            self.rel_tp / (self.rel_fn + self.rel_tp),
            self.rel_p_tp / (self.rel_p_fn + self.rel_p_tp)
        )

    def get_f1(self, average='micro'):
        return (
            (2*self.get_p()[0]*self.get_r()[0]) / (self.get_p()[0] + self.get_r()[0]), 
            (2*self.get_p()[1]*self.get_r()[1]) / (self.get_p()[1] + self.get_r()[1]), 
            (2*self.get_p()[2]*self.get_r()[2]) / (self.get_p()[2] + self.get_r()[2])
        )
    
    def get_prf(self, average='micro'):
        return self.get_p(), self.get_r(), self.get_f1()
    