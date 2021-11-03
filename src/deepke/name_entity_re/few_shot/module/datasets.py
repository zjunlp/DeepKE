import torch
from tqdm import tqdm
import numpy as np
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer

import logging
logger = logging.getLogger(__name__)


# load file and process bio
class ConllNERProcessor(object):
    def __init__(self, data_path, mapping, bart_name, learn_weights) -> None:
        self.data_path = data_path
        self.tokenizer = BartTokenizer.from_pretrained(bart_name)
        self.mapping = mapping  # 记录的是原始tag与转换后的tag的str的匹配关系
        self.original_token_nums = self.tokenizer.vocab_size
        self.learn_weights = learn_weights
        self._add_tags_to_tokens()

    def load_from_file(self, mode='train'):
        """load conll ner from file

        Args:
            mode (str, optional): train/test/dev. Defaults to 'train'.
        Return:
            outputs (dict)
            raw_words: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
            raw_targets: ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
            entities: [['EU'], ['German'], ['British']]
            entity_tags: ['org', 'misc', 'misc']
            entity_spans: [[0, 1], [2, 3], [6, 7]]
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        # extract bio
        split_c = '\t' if 'conll' in load_file  else ' '
        outputs = {'raw_words':[], 'raw_targets':[], 'entities':[], 'entity_tags':[], 'entity_spans':[]}
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            for line in lines:
                if line != "\n":
                    raw_word.append(line.split(split_c)[0])
                    raw_target.append(line.split(split_c)[1][:-1])
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        for words, targets in zip(raw_words, raw_targets):
            entities, entity_tags, entity_spans = [], [], []
            start, end, start_flag = 0, 0, False
            for idx, tag in enumerate(targets):
                if tag.startswith('B-'):    # 一个实体开头 另一个实体（I-）结束
                    end = idx
                    if start_flag:  # 另一个实体以I-结束，紧接着当前实体B-出现
                        entities.append(words[start:end])
                        entity_tags.append(targets[start][2:].lower())
                        entity_spans.append([start, end])
                        start_flag = False
                    start = idx
                    start_flag = True
                elif tag.startswith('I-'):  # 实体中间，不是开头也不是结束，end+1即可
                    end = idx
                elif tag.startswith('O'):  # 无实体，可能是上一个实体的结束
                    end = idx
                    if start_flag:  # 上一个实体结束
                        entities.append(words[start:end])
                        entity_tags.append(targets[start][2:].lower())
                        entity_spans.append([start, end])
                        start_flag = False
            if start_flag:  # 句子以实体I-结束，未被添加
                entities.append(words[start:end+1])
                entity_tags.append(targets[start][2:].lower())
                entity_spans.append([start, end+1])
                start_flag = False
    
            if len(entities) != 0:
                outputs['raw_words'].append(words)
                outputs['raw_targets'].append(targets)
                outputs['entities'].append(entities)
                outputs['entity_tags'].append(entity_tags)
                outputs['entity_spans'].append(entity_spans)
        return outputs

    def process(self, data_dict):
        target_shift = len(self.mapping) + 2 
        def prepare_target(item):
            raw_word = item['raw_word']
            word_bpes = [[self.tokenizer.bos_token_id]] 
            first = [] 
            cur_bpe_len = 1
            for word in raw_word:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                first.append(cur_bpe_len)
                cur_bpe_len += len(bpes)
                word_bpes.append(bpes)
            assert first[-1] + len(bpes) == sum(map(len, word_bpes))
            word_bpes.append([self.tokenizer.eos_token_id])
            assert len(first) == len(raw_word) == len(word_bpes) - 2

            lens = list(map(len, word_bpes)) 
            cum_lens = np.cumsum(lens).tolist()   

            entity_spans = item['entity_span']  # [(s1, e1, s2, e2), ()]
            entity_tags = item['entity_tag']  # [tag1, tag2...]
            entities = item['entity']  # [[ent1, ent2,], [ent1, ent2]]
            target = [0]
            pairs = []

            first = list(range(cum_lens[-1]))

            assert len(entity_spans) == len(entity_tags)                #
            for idx, (entity, tag) in enumerate(zip(entity_spans, entity_tags)):
                cur_pair = []
                num_ent = len(entity) // 2
                for i in range(num_ent):
                    start = entity[2 * i]
                    end = entity[2 * i + 1]
                    cur_pair_ = []
                    cur_pair_.extend([cum_lens[k] for k in list(range(start, end))])
                    cur_pair.extend([p + target_shift for p in cur_pair_])
                for _, (j, word_idx) in enumerate(zip((cur_pair[0], cur_pair[-1]), (0, -1))):
                    j = j - target_shift
                assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])

                cur_pair.append(self.mapping2targetid[tag] + 2) 
                pairs.append([p for p in cur_pair])
            target.extend(list(chain(*pairs)))
            target.append(1) 

            word_bpes = list(chain(*word_bpes))
            assert len(word_bpes)<500

            dict  = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first, 'src_seq_len':len(word_bpes), 'tgt_seq_len':len(target)}
            return dict
        
        logger.info("Process data...")
        for raw_word, raw_target, entity, entity_tag, entity_span in tqdm(zip(data_dict['raw_words'], data_dict['raw_targets'], data_dict['entities'], 
                                                                                data_dict['entity_tags'], data_dict['entity_spans']), total=len(data_dict['raw_words']), desc='Processing'):
            item_dict = prepare_target({'raw_word': raw_word, 'raw_target':raw_target, 'entity': entity, 'entity_tag': entity_tag, 'entity_span': entity_span})
            # add item_dict to data_dict
            for key, value in item_dict.items():
                if key in data_dict:
                    data_dict[key].append(value)
                else:
                    data_dict[key] = [value]
        return data_dict

    def _add_tags_to_tokens(self):
        mapping = self.mapping
        if self.learn_weights:  # add extra tokens to huggingface tokenizer
            self.mapping2id = {} 
            self.mapping2targetid = {} 
            for key, value in self.mapping.items():
                key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value[2:-2], add_prefix_space=True))
                self.mapping2id[value] = key_id  # may be list
                self.mapping2targetid[key] = len(self.mapping2targetid)
        else:
            tokens_to_add = sorted(list(mapping.values()), key=lambda x: len(x), reverse=True)  # 
            unique_no_split_tokens = self.tokenizer.unique_no_split_tokens                      # no split
            sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x: len(x), reverse=True)
            for tok in sorted_add_tokens:
                assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id    # 
            self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens          # add to no_split_tokens
            self.tokenizer.add_tokens(sorted_add_tokens)
            self.mapping2id = {}  # tag to id
            self.mapping2targetid = {}  # tag to number

            for key, value in self.mapping.items():
                key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
                assert len(key_id) == 1, value
                assert key_id[0] >= self.original_token_nums
                self.mapping2id[value] = key_id[0]  #
                self.mapping2targetid[key] = len(self.mapping2targetid)
        

class ConllNERDataset(Dataset):
    def __init__(self, data_processor, mode='train') -> None:
        self.data_processor = data_processor
        self.data_dict = data_processor.load_from_file(mode=mode)
        self.complet_data = data_processor.process(self.data_dict)
        self.mode = mode

    def __len__(self):
        return len(self.complet_data['src_tokens'])

    def __getitem__(self, index):
        if self.mode == 'test':
            return torch.tensor(self.complet_data['src_tokens'][index]), torch.tensor(self.complet_data['src_seq_len'][index]), \
                    torch.tensor(self.complet_data['first'][index]), self.complet_data['raw_words'][index]

        return torch.tensor(self.complet_data['src_tokens'][index]), torch.tensor(self.complet_data['tgt_tokens'][index]), \
                    torch.tensor(self.complet_data['src_seq_len'][index]), torch.tensor(self.complet_data['tgt_seq_len'][index]), \
                    torch.tensor(self.complet_data['first'][index]), self.complet_data['target_span'][index]


    def collate_fn(self, batch):
        src_tokens, src_seq_len, first  = [], [], []
        tgt_tokens, tgt_seq_len, target_span = [], [], []
        if self.mode == "test":
            raw_words = []
            for tup in batch:
                src_tokens.append(tup[0])
                src_seq_len.append(tup[1])
                first.append(tup[2])
                raw_words.append(tup[3])
            src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=self.data_processor.tokenizer.pad_token_id)
            first = pad_sequence(first, batch_first=True, padding_value=0)
            return src_tokens, torch.stack(src_seq_len, 0), first, raw_words

        for tup in batch:
            src_tokens.append(tup[0])
            tgt_tokens.append(tup[1])
            src_seq_len.append(tup[2])
            tgt_seq_len.append(tup[3])
            first.append(tup[4])
            target_span.append(tup[5])
        src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=self.data_processor.tokenizer.pad_token_id)
        tgt_tokens = pad_sequence(tgt_tokens, batch_first=True, padding_value=1)
        first = pad_sequence(first, batch_first=True, padding_value=0)
        return src_tokens, tgt_tokens, torch.stack(src_seq_len, 0), torch.stack(tgt_seq_len, 0), first, target_span


if __name__ == '__main__':
    data_path = {'train':'data/conll2003/train.txt'}
    bart_name = '../BARTNER-AMAX/facebook/'
    conll_processor = ConllNERProcessor(data_path, bart_name)
    conll_datasets = ConllNERDataset(conll_processor, mode='train')
    conll_dataloader = DataLoader(conll_datasets, collate_fn=conll_datasets.collate_fn, batch_size=8)
    for idx, data in enumerate(conll_dataloader):
        print(data)
        break
    
