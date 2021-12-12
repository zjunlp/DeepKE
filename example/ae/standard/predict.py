import os
import sys
import torch
import logging
import hydra
from hydra import utils
from deepke.attribution_extraction.standard.tools import Serializer
from deepke.attribution_extraction.standard.tools import _serialize_sentence, _convert_tokens_into_index, _add_pos_seq, _handle_attribute_data , _lm_serialize
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from deepke.attribution_extraction.standard.utils import load_pkl, load_csv
import deepke.attribution_extraction.standard.models as models




logger = logging.getLogger(__name__)


def _preprocess_data(data, cfg):
    attribute_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, 'attribute.csv'), verbose=False)
    atts = _handle_attribute_data(attribute_data)
    if cfg.model_name != 'lm':
        vocab = load_pkl(os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl'), verbose=False)
        cfg.vocab_size = vocab.count
        serializer = Serializer(do_chinese_split=cfg.chinese_split)
        serial = serializer.serialize

        _serialize_sentence(data, serial)
        _convert_tokens_into_index(data, vocab)
        _add_pos_seq(data, cfg)
        logger.info('start sentence preprocess...')
        formats = '\nsentence: {}\nchinese_split: {}\n' \
                'tokens:    {}\ntoken2idx: {}\nlength:    {}\nentity_index:  {}\nattribute_value_index:  {}'
        logger.info(
            formats.format(data[0]['sentence'], cfg.chinese_split,
                        data[0]['tokens'], data[0]['token2idx'], data[0]['seq_len'],
                        data[0]['entity_index'], data[0]['attribute_value_index']))
    else:
        _lm_serialize(data,cfg)
    return data, atts


def _get_predict_instance(cfg):
    flag = input('是否使用范例[y/n]，退出请输入: exit .... ')
    flag = flag.strip().lower()
    if flag == 'y' or flag == 'yes':
        sentence = '张冬梅，女，汉族，1968年2月生，河南淇县人，1988年7月加入中国共产党，1989年9月参加工作，中央党校经济管理专业毕业，中央党校研究生学历'
        entity = '张冬梅'
        attribute_value = '汉族'
    elif flag == 'n' or flag == 'no':
        sentence = input('请输入句子：')
        entity = input('请输入句中需要预测的实体：')
        attribute_value = input('请输入句中需要预测的属性值：')
    elif flag == 'exit':
        sys.exit(0)
    else:
        print('please input yes or no, or exit!')
        _get_predict_instance(cfg)
        

    instance = dict()
    instance['sentence'] = sentence.strip()
    instance['entity'] = entity.strip()
    instance['attribute_value'] = attribute_value.strip()
    instance['entity_offset'] = sentence.find(entity)
    instance['attribute_value_offset'] = sentence.find(attribute_value)

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

    x = dict()
    x['word'], x['lens'] = torch.tensor([data[0]['token2idx']]), torch.tensor([data[0]['seq_len']])
    
    if cfg.model_name != 'lm':
        x['entity_pos'], x['attribute_value_pos'] = torch.tensor([data[0]['entity_pos']]), torch.tensor([data[0]['attribute_value_pos']])
        if cfg.model_name == 'cnn':
            if cfg.use_pcnn:
                x['pcnn_mask'] = torch.tensor([data[0]['entities_pos']])
        if cfg.model_name == 'gcn':
            # 没找到合适的做 parsing tree 的工具，暂时随机初始化
            adj = torch.empty(1,data[0]['seq_len'],data[0]['seq_len']).random_(2)
            x['adj'] = adj


    for key in x.keys():
        x[key] = x[key].to(device)

    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.softmax(y_pred, dim=-1)[0]
        prob = y_pred.max().item()
        prob_att = list(rels.keys())[y_pred.argmax().item()]
        logger.info(f"\"{data[0]['entity']}\" 和 \"{data[0]['attribute_value']}\" 在句中属性为：\"{prob_att}\"，置信度为{prob:.2f}。")

    if cfg.predict_plot:
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
