from .ner_metric import NERMetric
from .re_metric import REMetric
from .ee_metric import EEMetric
from convert.utils.constant import NER, RE, EE, EEA, EET, KG

def get_metric(task):
    if task == NER or task == EET:
        return NERMetric
    elif task == RE or task == KG:
        return REMetric
    elif task == EE or task == EEA:
        return EEMetric
    else:
        raise ValueError("Invalid task: %s" % task)
    
