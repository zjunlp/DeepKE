from .ner_converter import NERConverter
from .re_converter import REConverter
from .spo_converter import SPOConverter
from .ee_converter import EEConverter
from .eea_converter import EEAConverter
from .eet_converter import EETConverter
from .kg_converter import KGConverter
from convert.utils.constant import NER, RE, SPO, EE, EET, EEA, KG


def get_converter(task):
    if task == NER:
        return NERConverter
    elif task == RE:
        return REConverter
    elif task == SPO:
        return SPOConverter
    elif task == EE:
        return EEConverter
    elif task == EET:
        return EETConverter
    elif task == EEA:
        return EEAConverter
    elif task == KG:
        return KGConverter
    else:
        raise KeyError