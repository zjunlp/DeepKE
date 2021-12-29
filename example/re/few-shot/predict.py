from logging import debug

import hydra
from hydra.utils import get_original_cwd

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import yaml
import time
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.optimization import get_linear_schedule_with_warmup
import os
from tqdm import tqdm

from deepke.relation_extraction.few_shot import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In order to ensure reproducible experiments, we must set random seeds.


def logging(log_dir, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_dir != '' and log_:
            with open(log_dir, 'a+') as f_log:
                f_log.write(s + '\n')

def test(args, model, lit_model, data):
    model.eval()
    with torch.no_grad():
        test_loss = []
        for test_index, test_batch in enumerate(tqdm(data.test_dataloader())):
            loss = lit_model.test_step(test_batch, test_index)
            test_loss.append(loss)
        f1 = lit_model.test_epoch_end(test_loss)
        logging(args.log_dir,
            '| test_result: {}'.format(f1))
        logging(args.log_dir,'-' * 89)



@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    cwd = get_original_cwd()
    os.chdir(cwd)
    if not os.path.exists(f"data/{cfg.model_name_or_path}.pt"):
        get_label_word(cfg)
    if not os.path.exists(cfg.data_dir):
        generate_k_shot(cfg.data_dir)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = REDataset(cfg)
    data_config = data.get_data_config()

    config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    config.num_labels = data_config["num_labels"]

    model = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path, config=config)

        
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
    

    model.to(device)

    lit_model = BertLitModel(args=cfg, model=model, device=device,tokenizer=data.tokenizer)
    data.setup()
    
    model.load_state_dict(torch.load(cfg.load_path)["checkpoint"], False)
    print("load trained model from {}.".format(cfg.load_path))

    test(cfg, model, lit_model, data)



if __name__ == "__main__":
    main()
