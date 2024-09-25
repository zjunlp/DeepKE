import os
import time
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch

import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from deepke.relation_extraction.document import *


def report(args, model, features):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(args, preds, features)
    return preds




@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    cwd = get_original_cwd()
    os.chdir(cwd)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(
        cfg.config_name if cfg.config_name else cfg.model_name_or_path,
        num_labels=cfg.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path,
    )

    Dataset = ReadDataset(cfg, cfg.dataset, tokenizer, cfg.max_seq_length)


    test_file = os.path.join(cfg.data_dir, cfg.test_file)

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
    #T_score, T_output = evaluate(cfg, model, T_features, tag="test")
    pred = report(cfg, model, T_features)
    with open("./result.json", "w") as fh:
        json.dump(pred, fh)


if __name__ == "__main__":
    main()