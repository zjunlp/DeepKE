from .chatglm import coll_fn_chatglm
from .moss import coll_fn_moss
from .llama import coll_fn_llama
from .cpmbee import coll_fn_cpmbee, DataCollatorForCPMBEE

COLL_FN_DICT = {
    "llama": coll_fn_llama,
    "moss": coll_fn_moss,
    "baichuan": coll_fn_llama,
    "chatglm": coll_fn_chatglm,
    "cpm-bee": coll_fn_cpmbee,
}