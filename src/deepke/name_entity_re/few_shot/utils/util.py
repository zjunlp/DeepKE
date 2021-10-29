import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer
import os

def avg_token_embeddings(tokenizer: BartTokenizer, bart_model: BartModel, bart_name, num_tokens):
    """when initial added tokens, use their averge token emebddings

    Args:
        tokenizer (BartTokenizer): [description]
        bart_model (BartModel): [description]
        bart_name ([type]): [description]
        num_tokens ([type]): [description]

    Raises:
        RuntimeError: [description]

    Returns:
        [type]: [description]
    """
    _tokenizer = BartTokenizer.from_pretrained(bart_name)
    for token in tokenizer.unique_no_split_tokens:
        if token[:2] == '<<':  # 特殊字符
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index)>1:
                raise RuntimeError(f"{token} wrong split")
            else:
                index = index[0]
            assert index>=num_tokens, (index, num_tokens, token)
            indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
            embed = bart_model.encoder.embed_tokens.weight.data[indexes[0]]
            for i in indexes[1:]:
                embed += bart_model.decoder.embed_tokens.weight.data[i]
            embed /= len(indexes)
            bart_model.decoder.embed_tokens.weight.data[index] = embed
    return bart_model


def seq_to_mask(seq_len, max_len):
    """[get attention mask with sequence length]

    Args:
        seq_len ([torch.tensor]): [shape: bsz, each sequence length in a batch]
    """
    max_len = int(max_len) if max_len else seq_len.max().long()
    cast_seq = torch.arange(max_len).expand(seq_len.size(0), -1).to(seq_len)
    mask = cast_seq.lt(seq_len.unsqueeze(1))
    return mask

def get_loss(tgt_tokens, tgt_seq_len, pred):
    """

    :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
    :param pred: bsz x max_len-1 x vocab_size
    :return:
    """
    tgt_seq_len = tgt_seq_len - 1
    mask = seq_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
    tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
    loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
    return loss

def get_model_device(model):
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device

def set_seed(seed=2021):
    """sets random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def convert_preds_to_outputs(preds, raw_words, mapping, tokenizer):
    """convet model predicitons to BIO outputs

    Args:
        preds ([torch.Tensor]): [prompt model predictions, (bsz x seq_len x labels)]
        raw_words ([List]): [source raw words]
        mapping ([dict]): [map entity labels to <<>>]
        tokenizer : [BartTokenizer]

    Returns:
        [outputs (List)]: [each item length equal to raw_words, BIO format.]
    """
    id2label = list(mapping.keys())
    pred_eos_index = preds.flip(dims=[1]).eq(1).cumsum(dim=1).long()
    preds = preds[:, 1:]
    pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
    pred_seq_len = (pred_seq_len - 2).tolist()

    word_start_index = len(mapping) + 2
    outputs = []
    for i, pred_item in enumerate(preds.tolist()):
        pred_item = pred_item[:pred_seq_len[i]] # single sentence prediction
        pairs, cur_pair = [], []
        if len(pred_item):  # this sentence prediciton= is not null
            for idx in pred_item:
                if idx < word_start_index:  # is entity
                    if len(cur_pair) > 0:
                        # assert word[i] < word[i+1]
                        if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                            pairs.append(tuple(cur_pair + [idx]))   # add valid words and current entity id
                    cur_pair = []   # clear word pairs
                else:   # is word
                    cur_pair.append(idx)    # add word id to word pairs
        raw_words_item = raw_words[i]
        cum_lens = [1]
        start_idx = 1
        for word in raw_words_item:
            start_idx += len(tokenizer.tokenize(word, add_prefix_space=True))
            cum_lens.append(start_idx)
        cum_lens.append(start_idx+1)
        output = ['O' for _ in range(len(raw_words_item))]
        # pairs: List[(word id, ... , entity id), (...), ...]
        for pair in pairs:  # (word id, ... , entity id)
            entity = pair[-1]
            words = []
            for word in pair[:-1]:
                if word-word_start_index in cum_lens:
                    words.append(cum_lens.index(word-word_start_index)) 
            if len(words) == 0: continue
            start_idx = words[0]
            end_idx = words[-1]
            output[start_idx] = f'B-{id2label[entity-2]}'
            for _ in range(start_idx+1, end_idx+1):
                output[_] = f'I-{id2label[entity-2]}'
        outputs.append(output)
    return outputs


def write_predictions(path, texts, labels):
    """[write model predictions to path (conll format)]

    Args:
        path ([str]): [save path]
        texts ([List]): [raw texts]
        labels ([List]): [predict labels]
    """
    print(len(texts), len(labels))
    assert len(texts) == len(labels)
    if not os.path.exists(path):
        os.system(r"touch {}".format(path))
    
    with open(path, "w", encoding="utf-8") as f:
        f.writelines("-DOCSTART-	O\n\n")
        for i in range(len(texts)):
            for j in range(len(texts[i])):
                f.writelines("{}\t{}\n".format(texts[i][j], labels[i][j]))
            f.writelines("\n")
