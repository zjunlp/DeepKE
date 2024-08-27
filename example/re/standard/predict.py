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
        sentence = '《穿越之鸣人传说》是玖血月所著的一部精彩纷呈的网络小说，这部作品自从在17k小说网连载以来，便受到了无数读者的关注和喜爱。这不仅因为它以广受欢迎的动漫《火影忍者》中的鸣人为原型展开了一段全新的穿越奇遇，也因为作者玖血月独到的想象力和扣人心弦的叙事技巧，使得这部作品呈现了与众不同的魅力。作品的主线围绕着主角鸣人的穿越经历展开，他意外穿越到了一个全新的世界，在这个世界中，他不仅要面对来自于未知世界的挑战，还要解开自己穿越的谜团。与原著动漫《火影忍者》中尚未展现的内容相比，这部小说更加注重个性化的冒险故事以及主角的内心成长。玖血月以高超的笔法，为读者构筑了一个既熟悉又陌生的奇幻世界。小说中，鸣人不仅保持了他乐观、坚韧不拔的性格特质，同时也因为穿越而获得了新的力量和智慧。随着故事的深入发展，鸣人与新世界中的人物交织出了复杂而又紧密的关系网络。这些人物个性鲜明，有的成为了他的朋友和伙伴，有的则成为了道路上的挑战者和敌人。在这一过程中，鸣人不仅要学会如何运用自己的新力量，更要学会理解和包容，以智慧和勇气解决面前的一切困难和挑战。玖血月通过对人物的深刻刻画以及对情节的巧妙安排，使得《穿越之鸣人传说》成为了一部情节与角色成长并重的作品。作者巧妙地将原著中的元素与全新创造的内容融合在一起，既保持了角色原有的魅力，又为其赋予了新的生命力。这不仅让熟悉原著的读者感到新鲜和惊喜，也能够吸引那些对原著不太了解的读者。'
        head = '穿越之鸣人传说'
        tail = '17k小说网'
        head_type = '网络小说'
        tail_type = '网站'
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
