import os
import time
import hydra
import numpy as np
import torch

import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from deepkeredoc import *

def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    total_loss = 0
    for i, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            output = model(**inputs)
            loss = output[0]
            pred = output[1].cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            total_loss += loss.item()

    average_loss = total_loss / (i + 1)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, re_p, re_r = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_re_p": re_p * 100,
        tag + "_re_r": re_r * 100,
        tag + "_average_loss": average_loss
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    preds = np.array(preds).astype(np.float32)
    preds = to_official(preds, features)
    return preds


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.n_gpu = torch.cuda.device_count()
    cfg.device = device

    config = AutoConfig.from_pretrained(
        cfg.config_name if cfg.config_name else cfg.model_name_or_path,
        num_labels=cfg.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path,
    )

    Dataset = ReadDataset(cfg.dataset, tokenizer, cfg.max_seq_length)

    train_file = os.path.join(cfg.data_dir, cfg.train_file)
    dev_file = os.path.join(cfg.data_dir, cfg.dev_file)
    test_file = os.path.join(cfg.data_dir, cfg.test_file)
    train_features = Dataset.read(train_file)
    dev_features = Dataset.read(dev_file)
    test_features = Dataset.read(test_file)

    model = AutoModel.from_pretrained(
        cfg.model_name_or_path,
        from_tf=bool(".ckpt" in cfg.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = cfg.transformer_type

    set_seed(cfg)
    model = DocREModel(config, cfg,  model, num_labels=cfg.num_labels)


    model.load_state_dict(torch.load(cfg.load_path)['checkpoint'])
    model.to(device)
    T_features = test_features  # Testing on the test set
    T_score, T_output = evaluate(cfg, model, T_features, tag="test")
    pred = report(cfg, model, T_features)
    print(pred)
    with open("./submit_result/result.json", "w") as fh:
        json.dump(pred, fh)


if __name__ == "__main__":
    main()