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

import wandb

def train(args, model, train_features, dev_features, test_features):
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_ and args.log_dir != '':
            with open(args.log_dir, 'a+') as f_log:
                f_log.write(s + '\n')
    def finetune(features, optimizer, num_epoch, num_steps, model):
        cur_model = model.module if hasattr(model, 'module') else model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.train_from_saved_model != '':
            best_score = torch.load(args.train_from_saved_model)["best_f1"]
            epoch_delta = torch.load(args.train_from_saved_model)["epoch"] + 1
        else:
            epoch_delta = 0
            best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = [epoch + epoch_delta for epoch in range(num_epoch)]
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        global_step = 0
        log_step = 100
        total_loss = 0
        


        #scaler = GradScaler()
        for epoch in train_iterator:
            start_time = time.time()
            optimizer.zero_grad()

            for step, batch in enumerate(train_dataloader):
                model.train()

                inputs = {'input_ids': batch[0].to(device),
                          'attention_mask': batch[1].to(device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                #with autocast():
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                total_loss += loss.item()
                #    scaler.scale(loss).backward()
               

                loss.backward()

                if step % args.gradient_accumulation_steps == 0:
                    #scaler.unscale_(optimizer)
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(cur_model.parameters(), args.max_grad_norm)
                    #scaler.step(optimizer)
                    #scaler.update()
                    #scheduler.step()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    num_steps += 1
                    if global_step % log_step == 0:
                        cur_loss = total_loss / log_step
                        elapsed = time.time() - start_time
                        logging(
                            '| epoch {:2d} | step {:4d} | min/b {:5.2f} | lr {} | train loss {:5.3f}'.format(
                                epoch, global_step, elapsed / 60, scheduler.get_last_lr(), cur_loss * 1000))
                        total_loss = 0
                        start_time = time.time()

                        wandb.log({
                            "train_loss":cur_loss
                        })

                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                # if step ==0:
                    logging('-' * 89)
                    eval_start_time = time.time()
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")

                    logging(
                        '| epoch {:3d} | time: {:5.2f}s | dev_result:{}'.format(epoch, time.time() - eval_start_time,
                                                                                dev_output))

                    wandb.log({
                            "dev_result":dev_output
                    })

                    logging('-' * 89)
                    if dev_score > best_score:
                        best_score = dev_score
                        logging(
                            '| epoch {:3d} | best_f1:{}'.format(epoch, best_score))

                        wandb.log({
                            "best_f1":best_score
                        })

                        if args.save_path != "":
                            torch.save({
                                'epoch': epoch,
                                'checkpoint': cur_model.state_dict(),
                                'best_f1': best_score,
                                'optimizer': optimizer.state_dict()
                            }, args.save_path
                            , _use_new_zipfile_serialization=False)
                            logging(
                                '| successfully save model at: {}'.format(args.save_path))
                            logging('-' * 89)
        return num_steps

    cur_model = model.module if hasattr(model, 'module') else model
    extract_layer = ["extractor", "bilinear"]
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
        {"params": [p for n, p in cur_model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.train_from_saved_model != '':
        optimizer.load_state_dict(torch.load(args.train_from_saved_model)["optimizer"])
        print("load saved optimizer from {}.".format(args.train_from_saved_model))
    

    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, model)


def evaluate(args, model, features, tag="dev"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    total_loss = 0
    for i, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
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
    ans = to_official(args, preds, features)
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


wandb.init(project="DeepKE_RE_Document")
wandb.watch_called = False
@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    cwd = get_original_cwd()
    os.chdir(cwd)

    if not os.path.exists(os.path.join(cfg.data_dir, "train_distant.json")):
        raise FileNotFoundError("Sorry, the file: 'train_annotated.json' is too big to upload to github, \
            please manually download to 'data/' from DocRED GoogleDrive https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(
        cfg.config_name if cfg.config_name else cfg.model_name_or_path,
        num_labels=cfg.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path,
    )
    
    Dataset = ReadDataset(cfg, cfg.dataset, tokenizer, cfg.max_seq_length)

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
    wandb.watch(model, log="all")


    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = cfg.transformer_type

    set_seed(cfg)
    model = DocREModel(config, cfg,  model, num_labels=cfg.num_labels)
    if cfg.train_from_saved_model != '':
        model.load_state_dict(torch.load(cfg.train_from_saved_model)["checkpoint"])
        print("load saved model from {}.".format(cfg.train_from_saved_model))
    
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
    model.to(device)

    train(cfg, model, train_features, dev_features, test_features)



if __name__ == "__main__":
    main()