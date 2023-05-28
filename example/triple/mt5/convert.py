import argparse
import re
import json

RE_triple = re.compile('(\([^\(\)]*\)?)')

def rte_post_process(result):    
    def clean(spt, i):
        try:
            s = spt[i].strip()
        except IndexError:
            s = ""
        if i == 0 and len(s) != 0 and s[0] == '(':
            s = s[1:]
        if i == 2 and len(s) != 0 and s[-1] == ')':
            s = s[:-1]
        return s

    matches = re.findall(RE_triple, result)
    new_record = []
    for m in matches:
        spt = m.split(",")
        head, rel, tail = clean(spt, 0), clean(spt, 1), clean(spt, 2)
        if head == "" or rel == "" or tail == "":
            continue
        new_record.append([head, rel, tail])

    return new_record

def clean(s):
    if s.startswith("输入中包含的关系三元组是："):
        return s.replace("输入中包含的关系三元组是：", "")
    return s


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--src_path", type=str, default="data/valid.json")
    parse.add_argument("--pred_path", type=str, default="output/ccks_mt5-base_f1_1e-4/test_preds.json")
    parse.add_argument("--tgt_path", type=str, default="output/valid_result.json")
    options = parse.parse_args()

    reader1 = open(options.src_path, "r", encoding="utf-8")
    reader2 = open(options.pred_path, "r", encoding="utf-8")
    writer = open(options.tgt_path, "w", encoding="utf-8")

    for line1, line2 in zip(reader1, reader2):
        data1 = json.loads(line1.strip())
        data2 = json.loads(line2.strip())

        data1['output'] = clean(data2['output'])
        data1['kg'] = rte_post_process(data2['output'])
        writer.write(json.dumps(data1, ensure_ascii=False) + "\n")
