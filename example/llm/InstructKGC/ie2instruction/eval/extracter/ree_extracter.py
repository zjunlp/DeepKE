import re
import json
from eval.extracter.extracter import Extracter


class REEExtracter(Extracter):
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
                subject = iit.get('subject', '')
                subject_type = iit.get('subject', '')
                object = iit.get('object', '')
                object_type = iit.get('object_type', '')
                if type(subject) != str or type(object) != str or type(subject_type) != str or type(object_type) != str:
                    print("type(subject) != str or type(object) != str or type(subject_type) != str or type(object_type) != str", result)
                    continue
                new_record.append((subject, subject_type, key, object, object_type))
        return True, new_record
        
