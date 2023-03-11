import argparse
import json
from collections import Counter, defaultdict
import os, sys
from tqdm import tqdm

# trigger_dict = {"O": 0}
# role_dict = {"O": 0}

def label_data(data, start, l, _type, task=None):
        """label_data"""
        for i in range(start, start + l):
            suffix = "B-" if i == start else "I-"
            label = "{}{}".format(suffix, _type)
            data[i] = label
            # if task == "trigger" and label not in trigger_dict.keys():
            #     trigger_dict[label] = len(trigger_dict) - 1
            # if task == "role" and label not in role_dict.keys():
            #     role_dict[label] = len(role_dict)
        return data

def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w", encoding="utf8") as outfile:
        print("len data:", len(data))
        [outfile.write(d + " \n") for d in data]



def data_process(raw_data, task="trigger", mode="train"):

    output = ["text_a\tlabel\tindex"] if task == "trigger" else ["text_a\tlabel\ttrigger_tag\tindex"]

    for idx, line in enumerate(tqdm(raw_data, desc="processing " + mode + " " + task)):
        line = json.loads(line)

        text_a = line["tokens"]
        if task == "trigger":
            trigger_labels = ["O"] * len(text_a)
            for mention in line["event_mentions"]:
                event_type = mention["event_type"]
                trigger_start = mention["trigger"]["start"]
                trigger_end = mention["trigger"]["end"]
                trigger = mention["trigger"]["text"]
                trigger_labels = label_data(trigger_labels, trigger_start, trigger_end - trigger_start, event_type, task)
            assert len(text_a) == len(trigger_labels)
            output.append("{}\t{}\t{}".format(" ".join(text_a), " ".join(trigger_labels), str(idx)))
        elif task == "role":

            # for negative samples
            if len(line["event_mentions"]) == 0:
                output.append("{}\t{}\t{}\t{}".format(" ".join(text_a), " ".join(["O"] * len(text_a)), " ".join(["O"] * len(text_a)), str(idx)))
                continue
            entity_mention = line["entity_mentions"]
            entity_id2mention = {mention["id"]: mention for mention in entity_mention}

            for mention in line["event_mentions"]:

                trigger_labels = ["O"] * len(text_a)
                event_type = mention["event_type"]
                trigger_start = mention["trigger"]["start"]
                trigger_end = mention["trigger"]["end"]
                trigger = mention["trigger"]["text"]
                trigger_labels = label_data(trigger_labels, trigger_start, trigger_end - trigger_start, event_type)

                role_labels = ["O"] * len(text_a)
                for arg in mention["arguments"]:
                    role_type = arg["role"]
                    argument = arg["text"]
                    role_entity = entity_id2mention[arg["entity_id"]]
                    role_start = role_entity["start"]
                    role_end = role_entity["end"]
                    role_labels = label_data(role_labels, role_start, role_end - role_start, role_type, task)
                assert len(text_a) == len(role_labels) == len(trigger_labels)
                output.append("{}\t{}\t{}\t{}".format(" ".join(text_a), " ".join(role_labels), " ".join(trigger_labels), str(idx)))

    print(len(output))
    return output

            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path", type=str, required=True)
    parser.add_argument("-c", "--bertcrf_path", type=str, required=True)
    args = parser.parse_args()

    print("processing for bertcrf......")

    for fold in ["train", "dev", "test"]:
        print(os.path.join(args.base_path, fold + ".w1.oneie.json"))
        raw_data = open(os.path.join(args.base_path, fold + ".w1.oneie.json"), "r").readlines()
        processed_trigger = data_process(raw_data=raw_data, task="trigger", mode=fold)
        processed_role = data_process(raw_data=raw_data, task="role", mode=fold)

        trigger_path = os.path.join(args.bertcrf_path, "trigger")
        role_path = os.path.join(args.bertcrf_path, "role")

        if not os.path.exists(trigger_path):
            os.makedirs(trigger_path)
        if not os.path.exists(role_path):
            os.makedirs(role_path)

        write_by_lines(os.path.join(trigger_path, fold + ".tsv"), processed_trigger)
        write_by_lines(os.path.join(role_path, fold + ".tsv"), processed_role)

    print("end processing for bertcrf......")



if __name__ == "__main__":
    main()
    # json.dump(trigger_dict, open("trigger_tag.json", "w"), ensure_ascii=False)
    # json.dump(role_dict, open("role_tag.json", "w"), ensure_ascii=False)