from .predict_parser import *
from .spotasoc_predict_parser import *
from .utils import *

decoding_format_dict = {
    'spotasoc': SpotAsocPredictParser,
}


def get_predict_parser(decoding_schema, label_constraint):
    return decoding_format_dict[decoding_schema](label_constraint=label_constraint)
