from .bert import BertForSequenceClassification
from .bert.modeling_bert import BertGetLabelWord, BertUseLabelWord, BertDecouple, BertForMaskedLM
from .gpt2 import GPT2DoubleHeadsModel
from .gpt2.modeling_gpt2 import GPT2UseLabelWord

from .roberta import RobertaForSequenceClassification 
from .roberta.modeling_roberta import RobertaUseLabelWord