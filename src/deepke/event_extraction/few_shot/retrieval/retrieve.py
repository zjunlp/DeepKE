import json
import os.path
from os import path
import sys
from retrieve_utils import init_es, create_index, del_index, search, add_data
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_name", default="ace05e_bart", type=str, required=True,
                    help="the dataset name")
parser.add_argument("--index_name", default="example_store", type=str, required=True,
                    help="the index name in in the elasticsearch")
parser.add_argument("--file_name", default="example_store.json", type=str, required=True,
                    help="the example store file")

args = parser.parse_args()
data_name = args.data_name
index_name = args.index_name
file_name = args.file_name

es = init_es()


def read_data_from_file(file_name):
    inputs = open('store/' + file_name, 'r')
    return json.load(inputs)


def construct_store():
    data = read_data_from_file(file_name=file_name)
    print("number of instances in store:", len(data.keys()))
    create_index(es, index_name)
    add_data(es, data, index_name)

def retrieve_knowledge(data_name="ace05ep_bart", k=8, index_name="simple_store", th=True):
    input_dir = "processed_data/" + data_name
    output_dir = 'processed_data/retrieved/' + data_name

    print("generating", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    threshold = 20

    print("k:", k)
    if th: print("threshold:", threshold)

    for fold in [
        "train",
        "dev"
        , "test"
        , "train.001"
        , "train.003"
        , "train.005"
        , "train.010"
        , "train.020"
        , "train.030"
    ]:
        outputs = open(path.join(output_dir, fold + ".w1.oneie.json"), "w")
        with open(path.join(input_dir, fold + ".w1.oneie.json"), "r") as inputs:
            print('---generating ' + fold + ' set---')
            data = inputs.readlines()
            for _, line in zip(tqdm(range(len(data))), data):
                line = json.loads(line)
                sent = line["sentence"]
                inject_set = []
                flag = True
                for res in search(es, sent, index_name)['hits']['hits']:

                    # aceplus跳过第一个检索到的结果
                    if flag and fold.find("train") and data_name.find("ace05ep"):
                        flag = False
                        continue

                    # 如果句子检索相同则跳过
                    if sent == res["_source"]["key"]: continue
                    if th and res["_score"] < threshold: break
                    ins = res["_source"]["key"]
                    event_list = list(set(res["_source"]["event_type"]))
                    inject_set.append((ins, event_list))
                    if len(inject_set) >= k: break
                line["retrieval"] = inject_set
                outputs.writelines(json.dumps(line) + '\n')
                outputs.flush()


if __name__ == '__main__':
    if not es.indices.exists(index_name):
        construct_store()
    retrieve_knowledge(data_name=data_name, k=8, index_name=index_name)
