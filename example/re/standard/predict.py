import os
import sys
import torch
import logging
import hydra
from hydra import utils
from deepke.relation_extraction.standard.tools import Serializer
from deepke.relation_extraction.standard.tools import _serialize_sentence, _convert_tokens_into_index, _add_pos_seq, _handle_relation_data , _lm_serialize
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from deepke.relation_extraction.standard.utils import load_pkl, load_csv
import deepke.relation_extraction.standard.models as models


logger = logging.getLogger(__name__)


def _preprocess_data(data, cfg):
    
    relation_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, 'relation.csv'), verbose=False)
    rels = _handle_relation_data(relation_data)

    if cfg.model_name != 'lm':
        vocab = load_pkl(os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl'), verbose=False)
        cfg.vocab_size = vocab.count
        serializer = Serializer(do_chinese_split=cfg.chinese_split)
        serial = serializer.serialize

        _serialize_sentence(data, serial, cfg)
        _convert_tokens_into_index(data, vocab)
        _add_pos_seq(data, cfg)
        logger.info('start sentence preprocess...')
        formats = '\nsentence: {}\nchinese_split: {}\nreplace_entity_with_type:  {}\nreplace_entity_with_scope: {}\n' \
                'tokens:    {}\ntoken2idx: {}\nlength:    {}\nhead_idx:  {}\ntail_idx:  {}'
        logger.info(
            formats.format(data[0]['sentence'], cfg.chinese_split, cfg.replace_entity_with_type,
                        cfg.replace_entity_with_scope, data[0]['tokens'], data[0]['token2idx'], data[0]['seq_len'],
                        data[0]['head_idx'], data[0]['tail_idx']))
    else:
        _lm_serialize(data,cfg)

    return data, rels


def _get_predict_instance(cfg):
    flag = input('是否使用范例[y/n]，退出请输入: exit .... ')
    flag = flag.strip().lower()
    if flag == 'y' or flag == 'yes':
        sentence = '歌曲《人生长路》出自刘德华国语专辑《男人的爱》，由李泉作词作曲，2001年出行发版'
        head = '男人的爱'
        tail = '人生长路'
        head_type = '所属专辑'
        tail_type = '歌曲'
    elif flag == 'n' or flag == 'no':
        sentence = input('请输入句子：')
        head = input('请输入句中需要预测关系的头实体：')
        head_type = input('请输入头实体类型：')
        tail = input('请输入句中需要预测关系的尾实体：')
        tail_type = input('请输入尾实体类型：')
    elif flag == 'exit':
        sys.exit(0)
    else:
        print('please input yes or no, or exit!')
        _get_predict_instance()

    instance = dict()
    instance['sentence'] = sentence.strip()
    instance['head'] = head.strip()
    instance['tail'] = tail.strip()
    if head_type.strip() == '' or tail_type.strip() == '':
        cfg.replace_entity_with_type = False
        instance['head_type'] = 'None'
        instance['tail_type'] = 'None'
    else:
        instance['head_type'] = head_type.strip()
        instance['tail_type'] = tail_type.strip()

    return instance

@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    # cwd = cwd[0:-5]
    cfg.cwd = cwd
    cfg.pos_size = 2 * cfg.pos_limit + 2
    print(cfg.pretty())

    # get predict instance
    instance = _get_predict_instance(cfg)
    data = [instance]

    # preprocess data
    data, rels = _preprocess_data(data, cfg)

    # model
    __Model__ = {
        'cnn': models.PCNN,
        'rnn': models.BiLSTM,
        'transformer': models.Transformer,
        'gcn': models.GCN,
        'capsule': models.Capsule,
        'lm': models.LM,
    }

    # 最好在 cpu 上预测
    cfg.use_gpu = False
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f'device: {device}')

    model = __Model__[cfg.model_name](cfg)
    logger.info(f'model name: {cfg.model_name}')
    logger.info(f'\n {model}')
    model.load(cfg.fp, device=device)
    model.to(device)
    model.eval()

    def process_single_piece(model, piece, device, rels):
        with torch.no_grad():
            for key in piece.keys():
                piece[key] = piece[key].to(device)
            y_pred = model(piece)
            y_pred = torch.softmax(y_pred, dim=-1)[0]  
            prob = y_pred.max().item()
            index = y_pred.argmax().item()
            if index >= len(rels):
                print("The index {} is out of range for 'rels' with length {}.".format(index, len(rels)))
                return [], 0, 0
            prob_rel = list(rels.keys())[index]
            return prob_rel, prob, y_pred
        
    if cfg.model_name != 'lm':
        x['word'], x['lens'] = torch.tensor([data[0]['token2idx'] + [0] * (512 - len(data[0]['token2idx']))]), torch.tensor([data[0]['seq_len']])
        x['head_pos'], x['tail_pos'] = torch.tensor(data[0]['head_pos'] + [0] * (512 - len(data[0]['token2idx']))), torch.tensor(data[0]['tail_pos'] + [0] * (512 - len(data[0]['token2idx'])))
        
        if cfg.model_name == 'cnn':
            if cfg.use_pcnn:
                x['pcnn_mask'] = torch.tensor([data[0]['entities_pos']])
        if cfg.model_name == 'gcn':
            # 没找到合适的做 parsing tree 的工具，暂时随机初始化
            adj = torch.empty(1,512,512).random_(2)
            x['adj'] = adj

        prob_rel, prob, y_pred = process_single_piece(model, x, device, rels)
        
        logger.info(f"\"{data[0]['head']}\" 和 \"{data[0]['tail']}\" 在句中关系为：\"{prob_rel}\"，置信度为{prob:.2f}。")
    
    else: 
        # 分片处理
        max_prob = -1
        best_relation = ''

        tokenized_input = data[0]['token2idx']
        max_len = 512  # 根据模型限制设置
        num_pieces = len(tokenized_input) // max_len + (1 if len(tokenized_input) % max_len > 0 else 0)
        
        for i in range(num_pieces):
            start_idx = i * max_len
            end_idx = min((i + 1) * max_len, len(tokenized_input))
            current_piece_input = {'word': torch.tensor([tokenized_input[start_idx:end_idx] + [0] * (max_len - (end_idx - start_idx))]),
                                'lens': torch.tensor([min(end_idx - start_idx, max_len)])}
            relation, prob, y_pred  = process_single_piece(model, current_piece_input, device, rels)
            if prob > max_prob:
                max_prob = prob
                best_relation = relation
        logger.info(f"\"{data[0]['head']}\" 和 \"{data[0]['tail']}\" 在句中关系为：\"{best_relation}\"，置信度为{max_prob:.2f}。")

    if cfg.predict_plot:
        # maplot 默认显示不支持中文
        plt.rcParams["font.family"] = 'Arial Unicode MS'
        x = list(rels.keys())
        height = list(y_pred.cpu().numpy())
        plt.bar(x, height)
        for x, y in zip(x, height):
            plt.text(x, y, '%.2f' % y, ha="center", va="bottom")
        plt.xlabel('关系')
        plt.ylabel('置信度')
        plt.xticks(rotation=315)
        plt.show()


if __name__ == '__main__':
    main()
