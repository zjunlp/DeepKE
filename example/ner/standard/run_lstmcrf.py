from torchcrf import CRF
import torch.nn as nn
from deepke.name_entity_re.standard import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, classification_report
import random
import sys
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

import wandb

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

wandb.init(project="DeepKE_NER_Standard")
@hydra.main(config_path="conf", config_name='config')


def main(cfg):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Use gpu or not
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')

    if os.path.exists(os.path.join(utils.get_original_cwd(), cfg.output_dir)) and cfg.model_save_name in os.listdir(os.path.join(utils.get_original_cwd(), cfg.output_dir)) and cfg.do_train:
        raise ValueError("Output Model ({}) already exists and is not empty.".format(os.path.join(utils.get_original_cwd(), cfg.output_dir, cfg.model_save_name)))
    if not os.path.exists(os.path.join(utils.get_original_cwd(), cfg.output_dir)):
        os.makedirs(os.path.join(utils.get_original_cwd(), cfg.output_dir))

    # Preprocess the dataset
    train_examples, word2id, label2id, id2label = build_crflstm_corpus('train', cfg)
    eval_examples = build_crflstm_corpus('dev', cfg)
    train_sampler = RandomSampler(train_examples)
    eval_sampler = SequentialSampler(eval_examples)

    # Prepare the model
    model = BiLSTM_CRF(len(word2id), len(label2id), cfg.word_dim, cfg.hidden_dim, cfg.drop_out, cfg.bidirectional, cfg.num_layers)
    model.to(device)

    train_dataloader = DataLoader(train_examples, sampler=train_sampler, batch_size=cfg.train_batch_size,
                                  collate_fn=lambda x: collate_fn(x, word2id, label2id))
    eval_dataloader = DataLoader(eval_examples, sampler=eval_sampler, batch_size=cfg.eval_batch_size,
                                 collate_fn=lambda x: collate_fn(x, word2id, label2id))

    # Define your optimizer & init crf parameters
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    scheduler = StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    for p in model.crf.parameters():
        _ = torch.nn.init.uniform_(p, -1, 1)

    if cfg.do_train:
        num_training_steps = cfg.num_train_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        max_avg_f1 = 0

        for epoch in trange(int(cfg.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            for step, batch in enumerate(train_dataloader):
                batch = (x.to(device) for x in batch)
                input, target, mask = batch
                y_pred = model(input, mask)
                loss = model.loss_fn(input, target, mask)
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()  # Update learning rate schedule
            progress_bar.update(len(train_dataloader))
            print(f'>>Epoch{epoch + 1}: Loss:{tr_loss / len(train_dataloader)}')


            with torch.no_grad():
                model.eval()
                y_true_list = []
                y_pred_list = []
                eval_loss = 0
                for step, batch in (enumerate(eval_dataloader)):
                    batch = (x.to(device) for x in batch)
                    input, target, mask = batch
                    y_pred = model(input, mask)
                    loss = model.loss_fn(input, target, mask)
                    if cfg.gradient_accumulation_steps > 1:
                        loss = loss / cfg.gradient_accumulation_steps

                    for lst in y_pred:
                        y_pred_list += lst
                    for y, m in zip(target, mask):
                        y_true_list += y[m == True].tolist()
                    eval_loss += loss.item()
                eval_labels = list(label2id.keys()).copy()
                eval_labels.remove('O')

                assert (np.array(y_true_list)).shape == (np.array(y_pred_list)).shape # Check whether the dimensions are equal
                assert (np.array(y_true_list)).ndim == 1    # Only accepts 1 dim
                assert (np.array(y_pred_list)).ndim == 1

                eval_f1 = f1_score([id2label[l] for l in y_true_list], [id2label[l] for l in y_pred_list],
                                   labels=eval_labels, average='weighted')
                print('>> total:', len(y_true_list), 'f1:', eval_f1, 'loss:', eval_loss / len(eval_dataloader))

                eval_loss /= len(eval_dataloader)

                if max_avg_f1 < eval_f1:
                    max_avg_f1 = eval_f1
                    print(f'model saved with eval_loss:{eval_loss}  f1:{eval_f1}')
                    torch.save(model, os.path.join(utils.get_original_cwd(), cfg.output_dir, cfg.model_save_name))


    if cfg.do_eval:
        evaluation_model = torch.load(os.path.join(utils.get_original_cwd(), cfg.output_dir, cfg.model_save_name), map_location='cpu')
        evaluation_model.to(device)
        evaluation_model.eval()

        evaluation_examples = build_crflstm_corpus(cfg.eval_on, cfg)

        evaluation_dataloader = DataLoader(eval_examples, batch_size=cfg.eval_batch_size,
                                     collate_fn=lambda x: collate_fn(x, word2id, label2id))
        y_true_list = []
        y_pred_list = []
        eval_loss = 0
        for step, batch in tqdm(enumerate(evaluation_dataloader), desc="Eval"):
            batch = (x.to(device) for x in batch)
            input, target, mask = batch
            y_pred = evaluation_model(input, mask)
            loss = evaluation_model.loss_fn(input, target, mask)
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

            for lst in y_pred:
                y_pred_list += lst
            for y, m in zip(target, mask):
                y_true_list += y[m == True].tolist()
            eval_loss += loss.item()

        report = classification_report([id2label[l] for l in y_true_list], [id2label[l] for l in y_pred_list],
                                       digits=4)
        print('>> total:', len(y_true_list), 'loss:', eval_loss / len(evaluation_dataloader))
        print(report)

if __name__ == '__main__':
    main()
