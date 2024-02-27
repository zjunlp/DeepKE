from .ee_extracter import EEExtracter
from .eet_extracter import EETExtracter
from .re_extracter import REExtracter
from .ner_extracter import NERExtracter
from .kg_extracter import KGExtracter
from convert.utils.constant import NER, RE, EE, EEA, EET, KG, MRC


def get_extracter(task):
    if task == NER:
        return NERExtracter
    elif task == RE:
        return REExtracter
    elif task == EE or task == EEA:
        return EEExtracter
    elif task == EET:
        return EETExtracter
    elif task == KG:
        return KGExtracter

    