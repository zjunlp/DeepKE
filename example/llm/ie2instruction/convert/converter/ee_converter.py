import json
from collections import defaultdict, OrderedDict
from convert.converter.converter import Converter

class EEConverter(Converter):
    def __init__(self, language="zh", NAN="NAN", prefix=""):
        super().__init__(language, NAN, prefix)

    def get_label_dict(self, records):
        labels = defaultdict(list)
        for event in records:
            event_trigger = event.get('event_trigger', '')
            event_type = event['event_type']
            if event_trigger == "" and len(event['arguments']) == 0:
                continue
            if event_trigger == "":
                event_trigger = self.NAN

            args = defaultdict(set)  # 论元可能多值, 但是也有唯一性
            for arg in event['arguments']:
                argument = arg['argument']
                role = arg['role']
                if argument != "":
                    args[role].add(argument)

            new_args = {}
            for role, arg in args.items():
                if len(arg) == 1:
                    new_args[role] = list(arg)[0]   # 只有一个值是一个字符串
                else:
                    new_args[role] = list(arg)  

            labels[event_type].append({"event_trigger":event_trigger, "arguments":new_args})
        return labels


    def convert_target(self, text, schema, label_dict):
        outputs = defaultdict(list)
        trigger = True
        for it in schema:
            trigger = it.get('trigger', True)
            if it['event_type'] not in label_dict:
                outputs[it['event_type']] = []
            else:
                args = []
                label_args = label_dict[it['event_type']]
                for label_arg in label_args:
                    tmp_args = OrderedDict()
                    flag = False
                    for role in it['arguments']:
                        if role not in label_arg["arguments"]:
                            tmp_args[role] = self.NAN
                        else:
                            flag = True
                            if isinstance(label_arg["arguments"][role], list):  # 多值论元, 按照在句子中的顺序排序
                                value_index = []
                                for value in label_arg["arguments"][role]:
                                    value_index.append([text.find(value), value])
                                value_index.sort(key=lambda x:x[0])
                                sorted_value = []
                                for iit in value_index:
                                    sorted_value.append(iit[1])
                                tmp_args[role] = sorted_value
                            tmp_args[role] = label_arg["arguments"][role]   # 只有一个值是一个字符串
                    if not flag:   #没有一个论元直接输出{}, 而非全部的NAN
                        tmp_args = {}
                    else:
                        tmp_args = dict(tmp_args)

                    if trigger:   # 需要添加触发词
                        args.append({"trigger":label_arg["event_trigger"], "arguments":tmp_args})
                    else:
                        if len(tmp_args) != 0:   # 不需要添加触发词, 但是有论元
                            args.append({"arguments":tmp_args})
                        # else: 不需要添加触发词, 也没有论元, 整个就是个空
                outputs[it['event_type']] = args

        if trigger:
            sorted_outputs = {}
            for key, value in outputs.items():
                value_index = []
                for it in value:
                    value_index.append([text.find(it['trigger']), it])
                value_index.sort(key=lambda x:x[0])
                sorted_value = []
                for it in value_index:
                    sorted_value.append(it[1])
                sorted_outputs[key] = sorted_value
        else:
            sorted_outputs = outputs
        output_text = json.dumps(dict(sorted_outputs), ensure_ascii=False)
        return self.prefix + output_text