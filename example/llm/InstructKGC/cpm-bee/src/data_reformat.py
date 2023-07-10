import json
import random

with open('train.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f]

num_data = len(data)
num_eval = int(num_data * 0.2)  # 20%作为验证集
eval_data = random.sample(data, num_eval)

with open('train.jsonl', 'w', encoding='utf-8') as f:
    for d in data:
        if d not in eval_data:
            cpm_d={"input":d["input"],"prompt":d["instruction"],"<ans>":d["output"]}
            json.dump(cpm_d, f, ensure_ascii=False)
            f.write('\n')

with open('eval.jsonl', 'w', encoding='utf-8') as f:
    for d in eval_data:
        cpm_d={"input":d["input"],"prompt":d["instruction"],"<ans>":d["output"]}
        json.dump(cpm_d, f, ensure_ascii=False)
        f.write('\n')