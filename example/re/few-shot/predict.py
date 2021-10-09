from logging import debug

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import yaml
import time
from lit_models import TransformerLitModelTwoSteps
from transformers import AutoConfig, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import os
from tqdm import tqdm

from deepkerefew import *

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = REDataset(cfg)
    data_config = data.get_data_config()

    config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    config.num_labels = data_config["num_labels"]

    model = BertForMaskedLM.from_pretrained(cfg.model_name_or_path, config=config)

    if cfg.load_path != '':
        model.load_state_dict(torch.load(cfg.load_path)["checkpoint"])
        print("load saved model from {}.".format(cfg.load_path))

        
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
    model.to(device)

    cur_model = model.module if hasattr(model, 'module') else model


    if "gpt" in cfg.model_name_or_path or "roberta" in cfg.model_name_or_path:
        tokenizer = data.get_tokenizer()
        cur_model.resize_token_embeddings(len(tokenizer))
        cur_model.update_word_idx(len(tokenizer))
        if "Use" in cfg.model_class:
            continous_prompt = [a[0] for a in tokenizer([f"[T{i}]" for i in range(1,3)], add_special_tokens=False)['input_ids']]
            continous_label_word = [a[0] for a in tokenizer([f"[class{i}]" for i in range(1, data.num_labels+1)], add_special_tokens=False)['input_ids']]
            discrete_prompt = [a[0] for a in tokenizer(['It', 'was'], add_special_tokens=False)['input_ids']]
            dataset_name = cfg.data_dir.split("/")[1]
            model.init_unused_weights(continous_prompt, continous_label_word, discrete_prompt, label_path=f"{cfg.model_name_or_path}_{dataset_name}.pt")

    lit_model = BertLitModel(cfg=cfg, model=model, tokenizer=data.tokenizer, device=device)
    data.setup()
    

    test(cfg, model, lit_model, data)



if __name__ == "__main__":
    main()
