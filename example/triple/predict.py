import hydra
import os
import torch
import logging
from hydra import utils
import json
from pyld import jsonld

from preprocess import  _handle_relation_data , _lm_serialize
from utils import  load_csv
from LMModel import LM
from InferBert import InferNer

logger = logging.getLogger(__name__)

def _preprocess_data(data, cfg):
    
    relation_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, 'relation.csv'), verbose=False)
    rels = _handle_relation_data(relation_data)
    _lm_serialize(data,cfg)
    return data, rels

def get_jsonld(head,rel,tail,url):
    
    doc = {
    "@id": head,
    url: {"@id":tail},
    }
    context = {
        rel:url
    }
    compacted = jsonld.compact(doc, context)
    logger.info(json.dumps(compacted, indent=2,ensure_ascii=False))

@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd

    label2word = {}
    with open(os.path.join(cfg.cwd, cfg.data_path, 'type.txt') , 'r') as f:
        data = f.readlines()
        for d in data:
            label2word[d.split(' ')[1].strip('\n')] = d.split(' ')[0]
    logger.info(label2word)
    rel2url = {}
    with open(os.path.join(cfg.cwd, cfg.data_path, 'url.txt') , 'r') as f:
        data = f.readlines()
        for d in data:
            rel2url[d.split(' ')[0]] = d.split(' ')[1].strip('\n')
    logger.info(rel2url)
    model = InferNer(cfg.nerfp)
    text = cfg.text

    logger.info(text)

    result = model.predict(text)
    logger.info(result)
    temp = ''
    last_type = result[0][1][2:]
    res = {}
    word_len = len(result)

    for i in range(word_len):
        k = result[i][0]
        v = result[i][1]
        
        if v[0] == 'B':
            if temp != '':
                res[temp] =  label2word[result[i - 1][1][2:]]
            temp = k
            last_type = result[i][1][2:]
        elif v[0] == 'I':
            if last_type == result[i][1][2:]:
                temp += k        
        

    if temp != '':
        res[temp] = label2word[result[len(result) - 1][1][2:]]
    logger.info(res)
    entity = []
    entity_label = []
    for k,v in res.items():
      entity.append(k)
      entity_label.append(v)
    
    entity_len = len(entity_label)

    for i in range(entity_len):
      for j in range(i + 1,entity_len):
        instance = dict()
        instance['sentence'] = text.strip()
        instance['head'] = entity[i].strip()
        instance['tail'] = entity[j].strip()
        instance['head_type'] = entity_label[i].strip()
        instance['tail_type'] = entity_label[j].strip()
    
        data = [instance]
        data, rels = _preprocess_data(data, cfg)

        device = torch.device('cpu')

        model = LM(cfg)
        model.load(cfg.refp, device=device)
        model.to(device)
        model.eval()

        x = dict()
        x['word'], x['lens'] = torch.tensor([data[0]['token2idx']]), torch.tensor([data[0]['seq_len']])
        for key in x.keys():
            x[key] = x[key].to(device)

        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=-1)[0]
            prob = y_pred.max().item()
            prob_rel = list(rels.keys())[y_pred.argmax().item()]
            logger.info(f"\"{data[0]['head']}\" 和 \"{data[0]['tail']}\" 在句中关系为：\"{prob_rel}\"，置信度为{prob:.2f}。")
            get_jsonld(data[0]['head'],prob_rel,data[0]['tail'],rel2url[prob_rel])

if __name__ == '__main__':
    main()