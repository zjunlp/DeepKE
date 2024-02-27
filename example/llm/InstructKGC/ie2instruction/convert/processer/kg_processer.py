import random
random.seed(42)
import numpy as np
np.random.seed(42)
from convert.processer.processer import Processer


class KGProcesser(Processer):
    def __init__(self, type_list, role_list, type_role_dict, negative=-1):
        super().__init__(type_list, role_list, type_role_dict, negative)
    