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

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"




def logging(log_dir, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_dir != '' and log_:
            with open(log_dir, 'a+') as f_log:
                f_log.write(s + '\n')


wandb.init(project="DeepKE_RE_Few")
wandb.watch_called = False
@hydra.main(config_path="./conf", config_name="config.yaml")
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
    wandb.watch(model, log="all")

    lit_model = BertLitModel(args=cfg, model=model, device=device, tokenizer=data.tokenizer)


    data.setup()
    
    if cfg.train_from_saved_model != '':
        model.load_state_dict(torch.load(cfg.train_from_saved_model)["checkpoint"])
        print("load saved model from {}.".format(cfg.train_from_saved_model))
        lit_model.best_f1 = torch.load(cfg.train_from_saved_model)["best_f1"]
    #data.tokenizer.save_pretrained('test')
    

    optimizer = lit_model.configure_optimizers()
    if cfg.train_from_saved_model != '':
        optimizer.load_state_dict(torch.load(cfg.train_from_saved_model)["optimizer"])
        print("load saved optimizer from {}.".format(cfg.train_from_saved_model))

    num_training_steps = len(data.train_dataloader()) // cfg.gradient_accumulation_steps * cfg.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps * 0.1, num_training_steps=num_training_steps)
    log_step = 100


    logging(cfg.log_dir,'-' * 89, print_=False)
    logging(cfg.log_dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' INFO : START TO TRAIN ', print_=False)
    logging(cfg.log_dir,'-' * 89, print_=False)

    for epoch in range(cfg.num_train_epochs):
        model.train()
        num_batch = len(data.train_dataloader())
        total_loss = 0
        log_loss = 0
        for index, train_batch in enumerate(tqdm(data.train_dataloader())):
            loss = lit_model.training_step(train_batch, index) / cfg.gradient_accumulation_steps
            total_loss += loss.item()
            log_loss += loss.item()
            loss.backward()
            if (index + 1) % cfg.accumulate_grad_batches == 0 or (index + 1) == num_batch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if log_step > 0 and (index+1) % log_step == 0:
                cur_loss = log_loss / log_step
                logging(cfg.log_dir, 
                    '| epoch {:2d} | step {:4d} | lr {} | train loss {:5.3f}'.format(
                        epoch, (index+1), scheduler.get_last_lr(), cur_loss * 1000)
                    , print_=False)
                log_loss = 0
        avrg_loss = total_loss / num_batch

        wandb.log({
                "train_loss": avrg_loss
            })

        logging(cfg.log_dir,
            '| epoch {:2d} | train loss {:5.3f}'.format(
                epoch, avrg_loss * 1000))
            
        model.eval()
        with torch.no_grad():
            val_loss = []
            for val_index, val_batch in enumerate(tqdm(data.val_dataloader())):
                loss = lit_model.validation_step(val_batch, val_index)
                val_loss.append(loss)
            f1, best, best_f1 = lit_model.validation_epoch_end(val_loss)
            logging(cfg.log_dir,'-' * 89)
            logging(cfg.log_dir,
                '| epoch {:2d} | dev_result: {}'.format(epoch, f1))
            logging(cfg.log_dir,'-' * 89)
            logging(cfg.log_dir,
                '| best_f1: {}'.format(best_f1))
            logging(cfg.log_dir,'-' * 89)

            wandb.log({
                "dev_result": f1,
                "best_f1":best_f1
            })
            
            if cfg.save_path != "" and best != -1:
                save_path = cfg.save_path
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict()
                }, save_path
                , _use_new_zipfile_serialization=False)
                logging(cfg.log_dir,
                    '| successfully save model at: {}'.format(save_path))
                logging(cfg.log_dir,'-' * 89)


if __name__ == "__main__":
    main()
