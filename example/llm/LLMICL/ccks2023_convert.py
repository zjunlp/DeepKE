import re
import json
import random
import argparse

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



def add_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--input_file", type=str, default="./data/valid.json", help="input file path")
    parse.add_argument("--output_file", type=str, default="./data/valid_input.json", help="output file path")
    parse.add_argument("--number", str=int, default=5, help="number of examples")
    parse.add_argument("--mode", str=int, default="train", choices=["train", "test"])
    options = parse.parse_args()
    return options


def convert_train(options):
    records = []
    with open(options.input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            records.append(data)
    random.shuffle(records)
    with open(options.output_file, "w") as writer:
        new_records = []
        for data in records[:options.number]:
            data["text"] = data["instruction"] + data["input"]
            data["labels"] = data["output"]
            new_records.append(data)
        json.dump(new_records, writer, ensure_ascii=False)


def convert_test(options):
    reader = open(options.input_path, "r", encoding="utf-8")
    writer = open(options.output_path, "w", encoding="utf-8")

    for line in reader:
        data = json.loads(line.strip())
        data['kg'] = rte_post_process(data['output'])
        writer.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    options = add_args()
    if options.mode == "train":
        convert_train(options)
    else:
        convert_test(options)
    

if __name__ == "__main__":
    main()
