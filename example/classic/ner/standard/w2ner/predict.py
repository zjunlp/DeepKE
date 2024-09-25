import os.path

from deepke.name_entity_re.standard.w2ner import *
import numpy as np
import hydra
from hydra import utils
import pickle
import torch

def trans_Dataset(config, examples: List[InputExample]):
    D = []
    for example in examples:
        span_infos = []
        sentence, label, d = example.text_a.split(' '), example.label, []
        if len(sentence) > config.max_seq_len:
            continue
        assert len(sentence) == len(label)
        i = 0
        while i < len(sentence):
            flag = label[i]
            if flag[0] == 'B':
                start_index = i
                i+=1
                while(i < len(sentence) and label[i][0] == 'I'):
                    i+=1
                d.append([start_index, i, flag[2:]])
            elif flag[0] == 'I':
                start_index = i
                i+=1
                while(i < len(sentence) and label[i][0] == 'I'):
                    i+=1
                d.append([start_index, i, flag[2:]])
            else:
                i+=1
        for s_e_flag in d:
            start_span, end_span, flag = s_e_flag[0], s_e_flag[1], s_e_flag[2]
            span_infos.append({'index': list(range(start_span, end_span)), 'type': flag})
        D.append({'sentence': sentence, 'ner': span_infos})

    return D

@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    config = type('Config', (), {})()
    for key in cfg.keys():
        config.__setattr__(key, cfg.get(key))


    print('*********Building the vocabulary(Need Your Training Set!!!!)*********')

    processor = NerProcessor()


    train_examples = processor.get_train_examples(os.path.join(utils.get_original_cwd(),
                                                               config.data_dir))
    train_data = trans_Dataset(config, train_examples)


    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)

    print('Building Done!\n')

    entity_text = config.text
    config.label_num = len(vocab.label2id)

    print('*********Loading the Final Model*********')
    model = Model(config)
    model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(utils.get_original_cwd(), config.save_path, 'pytorch_model.bin')))
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)

    print('Loading Done!\n')

    length = len([word for word in entity_text])
    tokens = [tokenizer.tokenize(word) for word in entity_text]
    pieces = [piece for pieces in tokens for piece in pieces]

    bert_inputs = tokenizer.convert_tokens_to_ids(pieces)

    bert_inputs = np.array([tokenizer.cls_token_id] + bert_inputs + [tokenizer.sep_token_id])
    pieces2word = np.zeros((length, len(bert_inputs)), dtype=np.bool)
    grid_mask2d = np.ones((length, length), dtype=np.bool)
    dist_inputs = np.zeros((length, length), dtype=np.int)
    sent_length = length

    if tokenizer is not None:
        start = 0
        for i, pieces in enumerate(tokens):
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start + len(pieces)))
            pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
            start += len(pieces)

    for k in range(length):
        dist_inputs[k, :] += k
        dist_inputs[:, k] -= k

    for i in range(length):
        for j in range(length):
            if dist_inputs[i, j] < 0:
                dist_inputs[i, j] = dis2idx[-dist_inputs[i, j]] + 9
            else:
                dist_inputs[i, j] = dis2idx[dist_inputs[i, j]]
    dist_inputs[dist_inputs == 0] = 19

    result = []
    with torch.no_grad():
        bert_inputs = torch.tensor([bert_inputs], dtype=torch.long).cuda()
        grid_mask2d = torch.tensor([grid_mask2d], dtype=torch.bool).cuda()
        dist_inputs = torch.tensor([dist_inputs], dtype=torch.long).cuda()
        pieces2word = torch.tensor([pieces2word], dtype=torch.bool).cuda()
        sent_length = torch.tensor([sent_length], dtype=torch.long).cuda()

        outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

        length = sent_length
        grid_mask2d = grid_mask2d.clone()
        outputs = torch.argmax(outputs, -1)
        ent_c, ent_p, ent_r, decode_entities = decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

        decode_entities = decode_entities[0]

        input_sentence = [word for word in entity_text]
        for ner in decode_entities:

            ner_indexes, ner_label = ner
            entity = ''.join([input_sentence[ner_index] for ner_index in ner_indexes])
            entity_label = vocab.id2label[ner_label]
            result.append((entity, entity_label))

    print("NER句子:")
    print(entity_text)
    print('NER结果:')
    print(result)


if __name__ == "__main__":
    main()
