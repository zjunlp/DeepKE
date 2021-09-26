"""Experiment-running framework."""
import argparse
import importlib
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_group.add_argument("--accelerator", default=None)
    trainer_group.add_argument("--accumulate_grad_batches", default=1)
    trainer_group.add_argument("--amp_backend", default='native')
    trainer_group.add_argument("--amp_level", default='O2')
    trainer_group.add_argument("--auto_lr_find", default=False)
    trainer_group.add_argument("--auto_scale_batch_size", default=False)
    trainer_group.add_argument("--auto_select_gpus", default=False)
    trainer_group.add_argument("--benchmark", default=False)
    trainer_group.add_argument("--check_val_every_n_epoch", default=1)
    trainer_group.add_argument("--checkpoint_callback", default=True)
    trainer_group.add_argument("--default_root_dir", default=None)
    trainer_group.add_argument("--deterministic", default=False)
    trainer_group.add_argument("--devices", default=None)
    trainer_group.add_argument("--distributed_backend", default=None)
    trainer_group.add_argument("--fast_dev_run", default=False)
    trainer_group.add_argument("--flush_logs_every_n_steps", default=100)
    trainer_group.add_argument("--gpus", default=None)
    trainer_group.add_argument("--gradient_clip_algorithm", default='norm')
    trainer_group.add_argument("--gradient_clip_val", default=0.0)
    trainer_group.add_argument("--ipus", default=None)
    trainer_group.add_argument("--limit_predict_batches", default=1.0)
    trainer_group.add_argument("--limit_test_batches", default=1.0)
    trainer_group.add_argument("--limit_train_batches", default=1.0)
    trainer_group.add_argument("--limit_val_batches", default=1.0)
    trainer_group.add_argument("--log_every_n_steps", default=50)
    trainer_group.add_argument("--log_gpu_memory", default=None)
    trainer_group.add_argument("--logger", default=True)
    trainer_group.add_argument("--max_epochs", default=None)
    trainer_group.add_argument("--max_steps", default=None)
    trainer_group.add_argument("--max_time", default=None)
    trainer_group.add_argument("--min_epochs", default=None)
    trainer_group.add_argument("--min_steps", default=None)
    trainer_group.add_argument("--move_metrics_to_cpu", default=False)
    trainer_group.add_argument("--multiple_trainloader_mode", default='max_size_cycle')
    trainer_group.add_argument("--num_nodes", default=1)
    trainer_group.add_argument("--num_processes", default=1)
    trainer_group.add_argument("--num_sanity_val_steps", default=2)
    trainer_group.add_argument("--overfit_batches", default=0.0)
    trainer_group.add_argument("--plugins", default=None)
    trainer_group.add_argument("--precision", default=32)
    trainer_group.add_argument("--prepare_data_per_node", default=True)
    trainer_group.add_argument("--process_position", default=0)
    trainer_group.add_argument("--profiler", default=None)
    trainer_group.add_argument("--progress_bar_refresh_rate", default=None)
    trainer_group.add_argument("--reload_dataloaders_every_epoch", default=False)
    trainer_group.add_argument("--reload_dataloaders_every_n_epochs", default=0)
    trainer_group.add_argument("--replace_sampler_ddp", default=True)
    trainer_group.add_argument("--resume_from_checkpoint", default=None)
    trainer_group.add_argument("--stochastic_weight_avg", default=False)
    trainer_group.add_argument("--sync_batchnorm", default=False)
    trainer_group.add_argument("--terminate_on_nan", default=False)
    trainer_group.add_argument("--tpu_cores", default=None)
    trainer_group.add_argument("--track_grad_norm", default=-1)
    trainer_group.add_argument("--truncated_bptt_steps", default=None)
    trainer_group.add_argument("--val_check_interval", default=1.0)
    trainer_group.add_argument("--weights_save_path", default=None)
    trainer_group.add_argument("--weights_summary", default='top')

    parser = argparse.ArgumentParser(add_help=False, parents=[parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--data_class", type=str, default="DIALOGUE")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--log_dir", default='', type=str)
    parser.add_argument("--save_path", default='', type=str)
    parser.add_argument("--train_from_saved_model", default='', type=str)
    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from tqdm import tqdm
def _get_relation_embedding(data):
    train_dataloader = data.train_dataloader()
    #! hard coded
    relation_embedding = [[] for _ in range(36)]
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()
    model = model.to(device)


    cnt = 0
    for batch in tqdm(train_dataloader):
        with torch.no_grad():
            #! why the sample in this case will cause errors
            if cnt == 416:
                continue
            cnt += 1
            input_ids, attention_mask, token_type_ids , labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state.detach().cpu()
            _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
            bs = input_ids.shape[0]
            mask_output = logits[torch.arange(bs), mask_idx] # [batch_size, hidden_size]
            

            labels = labels.detach().cpu()
            mask_output = mask_output.detach().cpu()
            assert len(labels[0]) == len(relation_embedding)
            for batch_idx, label in enumerate(labels.tolist()):
                for i, x in enumerate(label):
                    if x:
                        relation_embedding[i].append(mask_output[batch_idx])
    
    # get the mean pooling
    for i in range(36):
        if len(relation_embedding[i]):
            relation_embedding[i] = torch.mean(torch.stack(relation_embedding[i]), dim=0)
        else:
            relation_embedding[i] = torch.rand_like(relation_embedding[i-1])

    del model
    return relation_embedding


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


def main():
    parser = _setup_parser()
    args = parser.parse_args()


    #pl.seed_everything(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")
    
    data = data_class(args)
    data_config = data.get_data_config()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = data_config["num_labels"]

    

    # gpt no config?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.train_from_saved_model != '':
        #model.load_state_dict(torch.load(args.train_from_saved_model)["checkpoint"])
        print("load saved model from {}.".format(args.train_from_saved_model))

    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
    model.to(device)

    cur_model = model.module if hasattr(model, 'module') else model


    if "gpt" in args.model_name_or_path or "roberta" in args.model_name_or_path:
        tokenizer = data.get_tokenizer()
        cur_model.resize_token_embeddings(len(tokenizer))
        cur_model.update_word_idx(len(tokenizer))
        if "Use" in args.model_class:
            continous_prompt = [a[0] for a in tokenizer([f"[T{i}]" for i in range(1,3)], add_special_tokens=False)['input_ids']]
            continous_label_word = [a[0] for a in tokenizer([f"[class{i}]" for i in range(1, data.num_labels+1)], add_special_tokens=False)['input_ids']]
            discrete_prompt = [a[0] for a in tokenizer(['It', 'was'], add_special_tokens=False)['input_ids']]
            dataset_name = args.data_dir.split("/")[1]
            model.init_unused_weights(continous_prompt, continous_label_word, discrete_prompt, label_path=f"{args.model_name_or_path}_{dataset_name}.pt")
    # data.setup()
    # relation_embedding = _get_relation_embedding(data)
    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer, device=device)
    if args.train_from_saved_model != '':
        lit_model.best_f1 = torch.load(args.train_from_saved_model)["best_f1"]
    data.tokenizer.save_pretrained('test')
    data.setup()

    optimizer = lit_model.configure_optimizers()
    if args.train_from_saved_model != '':
        optimizer.load_state_dict(torch.load(args.train_from_saved_model)["optimizer"])
        print("load saved optimizer from {}.".format(args.train_from_saved_model))
    
    num_training_steps = len(data.train_dataloader()) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps * 0.1, num_training_steps=num_training_steps)
    log_step = 100


    logging(args.log_dir,'-' * 89, print_=False)
    logging(args.log_dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' INFO : START TO TRAIN ', print_=False)
    logging(args.log_dir,'-' * 89, print_=False)

    for epoch in range(args.num_train_epochs):
        model.train()
        num_batch = len(data.train_dataloader())
        total_loss = 0
        log_loss = 0
        for index, train_batch in enumerate(tqdm(data.train_dataloader())):
            loss = lit_model.training_step(train_batch, index)
            total_loss += loss.item()
            log_loss += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if log_step > 0 and (index+1) % log_step == 0:
                cur_loss = log_loss / log_step
                logging(args.log_dir, 
                    '| epoch {:2d} | step {:4d} | lr {} | train loss {:5.3f}'.format(
                        epoch, (index+1), scheduler.get_last_lr(), cur_loss * 1000)
                    , print_=False)
                log_loss = 0
        avrg_loss = total_loss / num_batch
        logging(args.log_dir,
            '| epoch {:2d} | train loss {:5.3f}'.format(
                epoch, avrg_loss * 1000))
            
        model.eval()
        with torch.no_grad():
            val_loss = []
            for val_index, val_batch in enumerate(tqdm(data.val_dataloader())):
                loss = lit_model.validation_step(val_batch, val_index)
                val_loss.append(loss)
            f1, best, best_f1 = lit_model.validation_epoch_end(val_loss)
            logging(args.log_dir,'-' * 89)
            logging(args.log_dir,
                '| epoch {:2d} | dev_result: {}'.format(epoch, f1))
            logging(args.log_dir,'-' * 89)
            logging(args.log_dir,
                '| best_f1: {}'.format(best_f1))
            logging(args.log_dir,'-' * 89)
            if args.save_path != "" and best != -1:
                file_name = f"{epoch}-Eval_f1-{best_f1:.2f}.pt"
                save_path = args.save_path + '/' + file_name
                torch.save({
                    'epoch': epoch,
                    'checkpoint': cur_model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict()
                }, save_path
                , _use_new_zipfile_serialization=False)
                logging(args.log_dir,
                    '| successfully save model at: {}'.format(save_path))
                logging(args.log_dir,'-' * 89)



    path = args.save_path + '/config'

    if not os.path.exists(path):
        os.mkdir(path)
    config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join(path, day_name)):
        os.mkdir(os.path.join(path, day_name))
    config = vars(args)
    config["path"] = path
    with open(os.path.join(os.path.join(path, day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

    test(args, model, lit_model, data)



if __name__ == "__main__":

    main()
