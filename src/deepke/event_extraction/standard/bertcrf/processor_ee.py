""" Event detection CRF finetuning: utilities to work with ACE 2005 & DuEE """

from __future__ import absolute_import, division, print_function
import json
import logging
import os
import sys
from copy import deepcopy
from io import open
from transformers import XLMRobertaTokenizer, BertTokenizer, RobertaTokenizer, AutoTokenizer
from .utils_ee import read_by_lines, write_by_lines, load_dict, label_data, process_remained_pred_trigger, clear_wrong_tokens

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, trigger_word_ids):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            trigger_word_ids: set token_type_ids of trigger tokens to 1, used in argument extraction task
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.trigger_word_ids = trigger_word_ids


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, token_type_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.token_type_ids = token_type_ids

class DataProcessor(object):

    def __init__(self, task_name, tokenizer):
        self.task_name = task_name
        self.tokenizer = tokenizer

    """Base class for data converters for multiple choice data sets."""

    def get_examples(self, file_path, set_type):
        """Gets a collection of `InputExample`s for the train/dev/test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    def process_dev_with_pred_trigger(self):
        """Process the dev set with pred trigger."""
        raise NotImplementedError()
    
    def process_test_with_pred_trigger(self):
        """Process the dev set with test trigger."""
        raise NotImplementedError()
    
    
class DUEEProcessor(DataProcessor):
    """Processor for the DuEE data set."""
    
    def get_examples(self, file_path, set_type):
        """See base class."""
        logger.info("LOOKING AT {} train".format(file_path))
        return self._create_examples(file_path, set_type)
    
    def data_process(self, path, trigger_pred_list, model="role"):

        output = ["text_a\tlabel\ttrigger_tag\tindex"]
        with open(path) as f:
            raw = f.readlines()
            assert len(raw) == len(trigger_pred_list)
            for idx, line, trigger_pred,in tqdm(zip(range(len(raw)), raw, trigger_pred_list)):

                d_json = json.loads(line.strip())
                text_a = ["，" if t == " " or t == "\n" or t == "\t" else t for t in list(d_json["text"].lower())]
                text_a_cleaned = text_a

                if model == "role":
                    for event in d_json.get("event_list", []):
                        event_type = event["event_type"]
                        trigger_labels = ["O"] * len(text_a)
                        trigger_start = event["trigger_start_index"]
                        trigger = event["trigger"]
                        trigger_labels = label_data(trigger_labels, trigger_start, len(trigger), event_type)

                        text_a_cleaned, trigger_labels_cleaned, trigger_start = clear_wrong_tokens(text_a, trigger_labels, self.tokenizer, len(trigger_pred))

                        for i in range(trigger_start, trigger_start + len(trigger)):
                            if i == len(trigger_pred): break
                            trigger_labels_cleaned[i] = trigger_pred[i]
                            trigger_pred[i] = "O"

                        role_labels = ["O"] * len(text_a)
                        for arg in event["arguments"]:
                            role_type = arg["role"]
                            argument = arg["argument"]
                            role_start = arg["argument_start_index"]
                            role_labels = label_data(role_labels, role_start, len(argument), role_type)

                        text_a_cleaned, role_labels_cleaned, _ = clear_wrong_tokens(text_a, role_labels, self.tokenizer, len(trigger_pred))

                        assert len(text_a_cleaned) == len(role_labels_cleaned) == len(trigger_labels_cleaned)
                        output.append("{}\t{}\t{}\t{}".format("\002".join(text_a_cleaned), "\002".join(role_labels_cleaned), "\002".join(trigger_labels_cleaned), idx))

                    # print("trigger_pred_after:", trigger_pred)
                    trigger_remained_labels_list = process_remained_pred_trigger(trigger_pred)
                    
                    for remained_labels in trigger_remained_labels_list:
                        assert len(text_a_cleaned) == len(remained_labels)
                        output.append("{}\t{}\t{}\t{}".format("\002".join(text_a_cleaned), "\002".join(["O"] * len(text_a_cleaned)), "\002".join(remained_labels), idx))
                
        return output    

    
    def process_dev_with_pred_trigger(self, args, raw_path, out_file):
        raw_path = raw_path + "/raw"
        trigger_pred_list = json.load(open(args.dev_trigger_pred_file, "r"))
        dev_role = self.data_process(os.path.join(raw_path, "duee_dev.json"), trigger_pred_list, "role")
        write_by_lines(os.path.join(args.data_dir, out_file), dev_role)
    
    def process_test_with_pred_trigger(self, args, raw_path, out_file):
        '''Do not consider the test set in duee.'''
        pass

    def get_labels(self, tag_path):
        """See base class."""
        tag_path = os.path.join(tag_path, self.task_name + "_tag.dict")
        return load_dict(tag_path).keys()
    
    def _create_examples(self, file_path, set_type):
        """Creates examples for the training and dev sets."""

        examples = []   

        for (idx, line) in enumerate(open(file_path).readlines()[1:]):

            e_id = "%s-%s" % (set_type, idx)

            datas = line.strip('\n').split('\t')
            words = datas[0].split('\002')
            labels = datas[1].split('\002')
            trigger_word_ids = None if self.task_name == "trigger" else datas[2].split('\002')

            examples.append(
                InputExample(
                    guid=e_id,
                    words=words,
                    labels=labels,
                    trigger_word_ids=trigger_word_ids
                )
            )

        return examples



class ACEProcessor(DataProcessor):

    """Processor for the DuEE data set."""
    
    def get_examples(self, file_path, set_type):
        """See base class."""
        logger.info("LOOKING AT {} train".format(file_path))
        return self._create_examples(file_path, set_type)
    

    def data_process(self, raw_data, trigger_pred_list):
        
        output = ["text_a\tlabel\ttrigger_tag\tindex"]

        assert len(raw_data) == len(trigger_pred_list)
        for idx, line in enumerate(tqdm(raw_data, desc="processing pred_trigger")):

            line = json.loads(line)
            text_a = line["tokens"]
            trigger_pred = trigger_pred_list[idx]

            assert len(trigger_pred) == len(text_a)

            if len(line["event_mentions"]) == 0:
                output.append("{}\t{}\t{}\t{}".format(" ".join(text_a), " ".join(["O"] * len(text_a)), " ".join(trigger_pred), idx))
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

                for i in range(trigger_start, trigger_end):
                    if i == len(trigger_pred): break
                    trigger_labels[i] = trigger_pred[i]
                    trigger_pred[i] = "O"

                role_labels = ["O"] * len(text_a)
                for arg in mention["arguments"]:
                    role_type = arg["role"]
                    argument = arg["text"]
                    role_entity = entity_id2mention[arg["entity_id"]]
                    role_start = role_entity["start"]
                    role_end = role_entity["end"]
                    role_labels = label_data(role_labels, role_start, role_end - role_start, role_type)

                assert len(text_a) == len(role_labels) == len(trigger_labels)
                output.append("{}\t{}\t{}\t{}".format(" ".join(text_a), " ".join(role_labels), " ".join(trigger_labels), idx))

            trigger_remained_labels_list = process_remained_pred_trigger(trigger_pred)
            for remained_labels in trigger_remained_labels_list:
                assert len(text_a) == len(remained_labels)
                output.append("{}\t{}\t{}\t{}".format(" ".join(text_a), " ".join(["O"] * len(text_a)), " ".join(remained_labels), idx))

        print(len(output))
        return output    

    def process_dev_with_pred_trigger(self, args, raw_path, out_file):
        trigger_pred_list = json.load(open(args.dev_trigger_pred_file, "r"))
        raw_path = raw_path + "/degree"
        raw_data = open(os.path.join(raw_path, "dev.w1.oneie.json"), "r").readlines()
        dev_role = self.data_process(raw_data, trigger_pred_list)
        write_by_lines(os.path.join(args.data_dir, out_file), dev_role)
        
    
    def process_test_with_pred_trigger(self, args, raw_path, out_file):
        trigger_pred_list = json.load(open(args.test_trigger_pred_file, "r"))
        raw_path = raw_path + "/degree"
        raw_data = open(os.path.join(raw_path, "test.w1.oneie.json"), "r").readlines()
        test_role = self.data_process(raw_data, trigger_pred_list)
        write_by_lines(os.path.join(args.data_dir, out_file), test_role)

    def get_labels(self, tag_path):
        """See base class."""
        tag_path = os.path.join(tag_path, self.task_name + "_tag.json")
        return json.load(open(tag_path, "r")).keys()
    
    def _create_examples(self, file_path, set_type):
        """Creates examples for the training and dev sets."""

        examples = []   

        for (idx, line) in enumerate(open(file_path).readlines()[1:]):

            e_id = "%s-%s" % (set_type, idx)

            datas = line.strip('\n').split('\t')
            words = datas[0].split()
            labels = datas[1].split()
            trigger_word_ids = None if self.task_name == "trigger" else datas[2].split()

            examples.append(
                InputExample(
                    guid=e_id,
                    words=words,
                    labels=labels,
                    trigger_word_ids=trigger_word_ids
                )
            )
        return examples



def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-100,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 model_name=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    # my logic in crf_padding requires this check. I create mask for crf by labels==pad_token_label_id to not include it
    # in loss and decoding
    assert pad_token_label_id not in label_map.values()

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("###############")
            logger.info("Writing example %d of %d", ex_index, len(examples))
            print("###############")

        tokens = []
        label_ids = []
        token_type_ids = []

        assert len(example.words) == len(example.labels)
        if example.trigger_word_ids is not None:
            assert len(example.words) == len(example.labels) == len(example.trigger_word_ids)

        # print("example.words:", example.words)
        # print("example.labels:", example.labels)
        # print("example.trigger_word_ids:", example.trigger_word_ids)

        for idx, word, label in zip(range(len(example.words)), example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            # clear out wrong chars
            if len(word_tokens) == 0: continue
            tokens.append(word)

            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if label!='X':
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len([word]) - 1))
            else:
                label_ids.extend([pad_token_label_id] + [pad_token_label_id] * (len([word]) - 1))

            if example.trigger_word_ids is not None:
                trigger_word_id = example.trigger_word_ids[idx]
                token_type = [0] if trigger_word_id == "O" else [1]
                token_type_ids.extend(token_type)
            else:
                token_type_ids.extend([0])
        
        # print("len(tokens):", len(tokens))
        # print("len(label_ids):", len(label_ids))
        # print("len(token_type_ids):", len(token_type_ids))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            token_type_ids = token_type_ids[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]  # [label_map["X"]]
        token_type_ids += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            token_type_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            token_type_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_type_ids += [0]
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        if model_name:
            if model_name == 'xlm-roberta-base':
                tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            elif model_name.startswith('bert'):
                tokenizer = BertTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            elif model_name == 'roberta':
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_type_ids = ([0] * padding_length) + token_type_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)
            token_type_ids += ([0] * padding_length)

        # print("len(input_ids):", len(input_ids))
        # print("len(label_ids):", len(label_ids))
        # print("len(token_type_ids):", len(token_type_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("token_type_ids: %s", " ".join([str(x) for x in token_type_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          token_type_ids=token_type_ids))
    return features


PROCESSORS = {
    "DuEE":DUEEProcessor, 
    "ACE":ACEProcessor
              }


if __name__ == "__main__":
    pad_token_label_id = -100
    tokenizer = AutoTokenizer.from_pretrained("../../models/bert-base-uncased")
    processor = PROCESSORS["ACE"]("role", tokenizer)
    # labels = processor.get_labels("./data/DuEE/schema/")
    processor.process_test_with_pred_trigger("./data/ACE/bertcrf/role", "./exp/ACE/trigger_bert-base-uncased/test_pred.json")


    # features = convert_examples_to_features(
    #         train_examples, 
    #         labels, 
    #         128, tokenizer,
    #         cls_token_at_end=bool("bertcrf" in ["xlnet"]),# xlnet has a cls token at the end
    #         cls_token=tokenizer.cls_token,
    #         cls_token_segment_id=2 if "bertcrf" in ["xlnet"] else 0,
    #         sep_token=tokenizer.sep_token,
    #         sep_token_extra=bool("bertcrf" in ["roberta"]),# roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
    #         pad_on_left=bool("bertcrf" in ["xlnet"]), # pad on the left for xlnet
    #         pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
    #         pad_token_segment_id=4 if "bertcrf" in ["xlnet"] else 0,
    #         pad_token_label_id=pad_token_label_id
    #     )

