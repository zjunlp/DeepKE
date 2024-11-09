from .ner_processer import NERProcesser
from .re_processer import REProcesser
from .spo_processer import SPOProcesser
from .ee_processer import EEProcesser
from .eet_processer import EETProcesser
from .eea_processer import EEAProcesser
from .kg_processer import KGProcesser

from convert.utils.constant import NER, RE, SPO, EE, EET, EEA, KG


def get_processer(task):
    if task == NER:
        return NERProcesser
    elif task == RE:
        return REProcesser
    elif task == SPO:
        return SPOProcesser
    elif task == EE:
        return EEProcesser
    elif task == EET:
        return EETProcesser
    elif task == EEA:
        return EEAProcesser
    elif task == KG:
        return KGProcesser
    else:
        raise KeyError