import re
import json
from eval.extracter.extracter import Extracter


class REExtracter(Extracter):
    def __init__(self, language="zh", NAN="NAN", prefix="输入中包含的关系三元组是：\n", Reject="No relation triples found."):
        super().__init__(language, NAN, prefix, Reject)


    def post_process(self, result):        
        try:      
            rst = json.loads(result)
        except json.decoder.JSONDecodeError:
            print("json decode error", result)
            return False, []
        if type(rst) != dict:
            print("type(rst) != dict", result)
            return False, []
        new_record = []
        for key, values in rst.items():
            if type(key) != str or type(values) != list:
                print("type(key) != str or type(values) != list", result)
                continue
            for iit in values:
                if type(iit) != dict:
                    print("type(iit) != dict", result)
                    continue
                head = iit.get('head', '')
                tail = iit.get('tail', '')
                if type(head) != str or type(tail) != str:
                    print("type(head) != str or type(tail) != str", result)
                    continue
                new_record.append((head, key, tail))
        return True, new_record


   