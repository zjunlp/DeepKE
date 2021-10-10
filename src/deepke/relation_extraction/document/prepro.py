from tqdm import tqdm
import torch.nn.functional as F
import ujson as json
import os
import pickle
import numpy as np
import torch


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention
    

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

class ReadDataset:
    def __init__(self, args, dataset: str, tokenizer, max_seq_Length: int = 1024,
             transformers: str = 'bert') -> None:
        self.transformers = transformers
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_Length = max_seq_Length
        self.args = args

    def read(self, file_in: str):
        save_file = file_in.split('.json')[0] + '_' + self.transformers + '_' \
                        + self.dataset + '.pkl'
        if self.dataset == 'docred':
            return read_docred(self.args, self.transformers, file_in, save_file, self.tokenizer, self.max_seq_Length)
        else:
            raise RuntimeError("No read func for this dataset.")



def read_docred(args, transfermers, file_in, save_file, tokenizer, max_seq_length=1024):
    if os.path.exists(save_file):
        with open(file=save_file, mode='rb') as fr:
            features = pickle.load(fr)
            fr.close()
        print('load preprocessed data from {}.'.format(save_file))
        return features
    else:
        max_len = 0
        up512_num = 0
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        features = []
        docred_rel2id = json.load(open(f'{args.data_dir}/rel2id.json', 'r'))
        if file_in == "":
            return None
        with open(file_in, "r") as fh:
            data = json.load(fh)
        if transfermers == 'bert':
            # entity_type = ["ORG", "-",  "LOC", "-",  "TIME", "-",  "PER", "-", "MISC", "-", "NUM"]
            entity_type = ["-", "ORG", "-",  "LOC", "-",  "TIME", "-",  "PER", "-", "MISC", "-", "NUM"]


        for sample in tqdm(data, desc="Example"):
            sents = []
            sent_map = []

            entities = sample['vertexSet']
            entity_start, entity_end = [], []
            mention_types = []
            for entity in entities:
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0]))
                    entity_end.append((sent_id, pos[1] - 1))
                    mention_types.append(mention['type'])

            for i_s, sent in enumerate(sample['sents']):
                new_map = {}
                for i_t, token in enumerate(sent):
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if (i_s, i_t) in entity_start:
                        t = entity_start.index((i_s, i_t))
                        if transfermers == 'bert':
                            mention_type = mention_types[t]
                            special_token_i = entity_type.index(mention_type)
                            special_token = ['[unused' + str(special_token_i) + ']']
                        else:
                            special_token = ['*']
                        tokens_wordpiece = special_token + tokens_wordpiece
                        # tokens_wordpiece = ["[unused0]"]+ tokens_wordpiece

                    if (i_s, i_t) in entity_end:
                        t = entity_end.index((i_s, i_t))
                        if transfermers == 'bert':
                            mention_type = mention_types[t]
                            special_token_i = entity_type.index(mention_type) + 50
                            special_token = ['[unused' + str(special_token_i) + ']']
                        else:
                            special_token = ['*']
                        tokens_wordpiece = tokens_wordpiece + special_token

                        # tokens_wordpiece = tokens_wordpiece + ["[unused1]"]
                        # print(tokens_wordpiece,tokenizer.convert_tokens_to_ids(tokens_wordpiece))

                    new_map[i_t] = len(sents)
                    sents.extend(tokens_wordpiece)
                new_map[i_t + 1] = len(sents)
                sent_map.append(new_map)

            if len(sents)>max_len:
                max_len=len(sents)
            if len(sents)>512:
                up512_num += 1

            train_triple = {}
            if "labels" in sample:
                for label in sample['labels']:
                    evidence = label['evidence']
                    r = int(docred_rel2id[label['r']])
                    if (label['h'], label['t']) not in train_triple:
                        train_triple[(label['h'], label['t'])] = [
                            {'relation': r, 'evidence': evidence}]
                    else:
                        train_triple[(label['h'], label['t'])].append(
                            {'relation': r, 'evidence': evidence})

            entity_pos = []
            for e in entities:
                entity_pos.append([])
                mention_num = len(e)
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))


            relations, hts = [], []
            # Get positive samples from dataset
            for h, t in train_triple.keys():
                relation = [0] * len(docred_rel2id)
                for mention in train_triple[h, t]:
                    relation[mention["relation"]] = 1
                    evidence = mention["evidence"]
                relations.append(relation)
                hts.append([h, t])
                pos_samples += 1

            # Get negative samples from dataset
            for h in range(len(entities)):
                for t in range(len(entities)):
                    if h != t and [h, t] not in hts:
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        relations.append(relation)
                        hts.append([h, t])
                        neg_samples += 1

            assert len(relations) == len(entities) * (len(entities) - 1)

            if len(hts)==0:
                print(len(sent))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            i_line += 1
            feature = {'input_ids': input_ids,
                       'entity_pos': entity_pos,
                       'labels': relations,
                       'hts': hts,
                       'title': sample['title'],
                       }
            features.append(feature)



        print("# of documents {}.".format(i_line))
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
        print("# {} examples len>512 and max len is {}.".format(up512_num, max_len))


        with open(file=save_file, mode='wb') as fw:
            pickle.dump(features, fw)
        print('finish reading {} and save preprocessed data to {}.'.format(file_in, save_file))

        return features