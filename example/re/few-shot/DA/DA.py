import os
import json
from collections import deque
from tqdm import tqdm
import argparse
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action

from nlpcda import Similarword


def read_dataset(fname):
    dataset = []
    with open(fname, 'r') as f:
        for line in tqdm(f.readlines(), desc='reading dataset'):
            dataset.append(json.loads(line))
    return dataset

def merge(sent_dict):
    # Merge sentence parts in order
    sent_order = ['sent1', 'ent1', 'sent2', 'ent2', 'sent3']
    q = deque([[sent] for sent in sent_dict[sent_order[0]]])
    curr_idx = 1
    while curr_idx < len(sent_order):
        curr_len = len(q)
        for _ in range(curr_len):
            prev_sents = q.pop()
            for postfix in sent_dict[sent_order[curr_idx]]:
                q.appendleft(prev_sents + [postfix])
        curr_idx += 1
    return list(q)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True,
                        help='the training set file')
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="The directory of the sampled files.")
    parser.add_argument("--language", "-lan", type=str, default="en", choices=["en","cn"],
                        help="DA for English or Chinese")
    parser.add_argument('--locations', '-l', nargs='+',
                        choices=['sent1', 'sent2', 'sent3', 'ent1', 'ent2'],
                        default=['sent1', 'sent2', 'sent3', 'ent1', 'ent2'],
                        help='List of positions that you want to manipulate')
    
    # DA for English
    parser.add_argument('--DAmethod', '-d', type=str, default="word_embedding_roberta",
                        choices=["word2vec","TF-IDF","word_embedding_bert","word_embedding_roberta","random_swap","synonym"],
                        help='Data augmentation method')
    parser.add_argument('--model_dir','-m', type=str,
                        help="the path of pretrained models used in DA methods",
                        default="./model")
    parser.add_argument("--model_name", "-mn", type=str, default="roberta-large",
                        help="model from huggingface")

    # DA for Chinese
    parser.add_argument("--create_num", "-cn", type=int, default=1,
                        help="The number of samples augmented from one instance.")
    parser.add_argument("--change_rate", "-cr", type=float, default=0.5,
                        help="the changing rate of text")

    args = parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    if args.language=="en":
        DAmethods = {
            "word2vec": '''naw.WordEmbsAug(
                        model_type='word2vec', 
                        model_path=os.path.join(args.model_dir,"GoogleNews-vectors-negative300.bin"),
                        action="substitute")''',
            "TF-IDF": '''naw.TfIdfAug(
                        model_path=args.model_dir,
                        action="substitute"
                        )''',
            "word_embedding_bert": '''naw.ContextualWordEmbsAug(
                                model_path=args.model_name, 
                                action="substitute",device='cuda')''',
            "word_embedding_roberta": '''naw.ContextualWordEmbsAug(
                                model_path="roberta-base",
                                action="substitute",device='cuda')''',
            "synonym": '''naw.SynonymAug(aug_src='wordnet')''',
            "random_swap": '''naw.RandomWordAug(action="swap")''',
            "synonym": '''naw.SynonymAug(aug_src='wordnet')'''
        }
        origin_data = read_dataset(args.input_file)
        DA_data = []
        replaced_samples = []
        perturb_func = eval(DAmethods[args.DAmethod])
        for example in tqdm(origin_data,desc="Augment the dataset"):
            tokens = example['token']
            relation = example['relation']
            head_pos, tail_pos = example['h']['pos'], example['t']['pos']
            rev = head_pos[0] > tail_pos[0]
            if rev:
                head_pos = example['t']['pos']
                tail_pos = example['h']['pos']
            # Split the tokens
            sent1, ent1, sent2, ent2, sent3 = (' '.join(tokens[:head_pos[0]]),
                                            ' '.join(
                                                tokens[head_pos[0]:head_pos[1]]),
                                            ' '.join(
                                                tokens[head_pos[1]:tail_pos[0]]),
                                            ' '.join(
                                                tokens[tail_pos[0]:tail_pos[1]]),
                                            ' '.join(tokens[tail_pos[1]:]))
            # Pack all parts into a dict and modify by names
            sent_dict = {'sent1': [sent1], 'ent1': [ent1], 'sent2': [sent2],
                        'ent2': [ent2], 'sent3': [sent3]}
            sent_dict_copy = sent_dict.copy()
            # Diverge
            for loc in args.locations:
                origin = sent_dict[loc][0]
                if not origin:
                    # No tokens given
                    continue
                ret = perturb_func.augment(origin)

                # Process result
                if not ret:
                    # Returned nothing
                    ret = [sent_dict[loc][0]]
                if isinstance(ret, str):
                    # Wrap single sentence
                    ret = [ret]
                sent_dict_copy[loc] = ret

            # Merge all parts of perturbed sentences and filter out original sentence
            for merged_sent in filter(lambda perturbed_tokens: perturbed_tokens != tokens,
                                    merge(sent_dict_copy)):
                tokens = ' '.join(merged_sent).split(' ')
                sent1, ent1, sent2, ent2, sent3 = merged_sent
                head_pos = [len(sent1.split(' '))]
                head_pos.append(head_pos[0] + len(ent1.split(' ')))
                tail_pos = [head_pos[1] + len(sent2.split(' '))]
                tail_pos.append(tail_pos[0] + len(ent2.split(' ')))
                if rev:
                    head_pos, tail_pos = tail_pos, head_pos
                replaced_samples.append({
                    'text': tokens,
                    'h': {'name':ent1, 'pos': head_pos},
                    't': {'name':ent2, 'pos': tail_pos},
                    'relation': relation,
                    'aug': args.DAmethod
                })
    else:
        # 中文DA
        origin_data = read_dataset(args.input_file)
        DA_data = []
        replaced_samples = []
        perturb_func = Similarword(create_num=1, change_rate=0.3)
        for example in tqdm(origin_data,desc="Augment the dataset"):
            tokens = example['text']
            relation = example['relation']
            head_pos, tail_pos = example['h']['pos'], example['t']['pos']
            rev = head_pos[0] > tail_pos[0]
            if rev:
                head_pos = example['t']['pos']
                tail_pos = example['h']['pos']
            # Split the tokens
            sent1, ent1, sent2, ent2, sent3 = (tokens[:head_pos[0]],
                                                tokens[head_pos[0]:head_pos[1]],
                                                tokens[head_pos[1]:tail_pos[0]],
                                                tokens[tail_pos[0]:tail_pos[1]],
                                                tokens[tail_pos[1]:])

            # Pack all parts into a dict and modify by names
            sent_dict = {'sent1': [sent1], 'ent1': [ent1], 'sent2': [sent2],
                        'ent2': [ent2], 'sent3': [sent3]}
            sent_dict_copy = sent_dict.copy()
            # Diverge
            for loc in args.locations:
                origin = sent_dict[loc][0]
                if not origin:
                    # No tokens given
                    continue
                ret = perturb_func.replace(origin)

                # Process result
                if not ret:
                    # Returned nothing
                    ret = [sent_dict[loc][0]]
                if isinstance(ret, str):
                    # Wrap single sentence
                    ret = [ret]
                sent_dict_copy[loc] = ret

            # Merge all parts of perturbed sentences and filter out original sentence
            for merged_sent in filter(lambda perturbed_tokens: perturbed_tokens != tokens,
                                    merge(sent_dict_copy)):
                tokens = merged_sent
                sent1, ent1, sent2, ent2, sent3 = merged_sent
                head_pos = [len(sent1)]
                head_pos.append(head_pos[0] + len(ent1))
                tail_pos = [head_pos[1] + len(sent2)]
                tail_pos.append(tail_pos[0] + len(ent2))
                if rev:
                    head_pos, tail_pos = tail_pos, head_pos
                replaced_samples.append({
                    'token': tokens,
                    'h': {'name':ent1, 'pos': head_pos},
                    't': {'name':ent2, 'pos': tail_pos},
                    'relation': relation,
                    'aug': "nlpcda_similarword"
                })
    
    with open(os.path.join(args.output_dir, "aug.txt"),'w') as f:
        for line in replaced_samples:
            f.writelines(json.dumps(line, ensure_ascii=False))
            f.write('\n')
