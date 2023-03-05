"""
    Runner for training and testing models.
    Tianyu Liu
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import logging
import random
import numpy as np
import torch

from torch import nn

from torch.optim import AdamW
from util.multigpu_fused_adam import FusedAdam

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import time
from os.path import join
from datetime import datetime

import util
from util.tensorize_coref import CorefDataProcessor, coref_collate_fn
from util.tensorize_ner import NERDataProcessor, ner_collate_fn
from util.tensorize_ere import EREDataProcessor, ere_collate_fn

from metrics import CorefEvaluator, MentionEvaluator

from models.model_coref import CorefModel
from models.model_ner import NERModel
from models.model_ere import EREModel

import wandb
wandb.init(project="DeepKE_TRIPLE_ASP")
wandb.watch_called = False
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)

class Runner:
    def __init__(self, config_file, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed
        # Set up seed
        if seed:
            util.set_seed(seed)
        # Set up config
        self.config = util.initialize_config(config_name, config_file=config_file)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up device
        self.device = 'cpu' if gpu_id is None else gpu_id
        # Use mixed precision training
        self.use_amp = self.config['use_amp']

        # Set up data
        if self.config['task'] == 'coref':
            self.data = CorefDataProcessor(self.config)
            self.collate_fn = coref_collate_fn
            self.model_class_fn = CorefModel
        elif self.config['task'] == 'ner':
            self.data = NERDataProcessor(self.config)
            self.collate_fn = ner_collate_fn
            self.model_class_fn = NERModel
        elif self.config['task'] == 'ere':
            self.data = EREDataProcessor(self.config)
            self.collate_fn = ere_collate_fn
            self.model_class_fn = EREModel


    def initialize_model(self, saved_suffix=None, continue_training=False):
        model = self.model_class_fn(self.config, self.device)
        
        if saved_suffix:
            model, start_epoch = self.load_model_checkpoint(model, saved_suffix, continue_training=continue_training)
            return model, start_epoch

        return model, 0


    def train(self, model, continued=False, start_epoch=0):
        logger.info('Config:')
        for name, value in self.config.items():
            logger.info('%s: %s' % (name, value))
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        epochs, grad_accum = self.config['num_epochs'], self.config['gradient_accumulation_steps']
        batch_size = self.config['batch_size']

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // (grad_accum * batch_size)
        if not continued:
            self.optimizer = self.get_optimizer(model)
            self.scheduler = self.get_scheduler(self.optimizer, total_update_steps)

        # Get model parameters for grad clipping
        plm_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)
        logger.info('Starting step: %d' % self.scheduler._step_count)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1, max_f1_test = 0, 0
        start_time = time.time()
        if type(self.optimizer) == FusedAdam:
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)

        trainloader = DataLoader(
            examples_train, batch_size=batch_size, shuffle=True, 
            num_workers=0,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

        for epo in range(start_epoch, epochs):
            logger.info("*******************EPOCH %d*******************" % epo)
            for doc_key, example in trainloader:
                # Forward pass
                model.train()
                example_gpu = {}
                for k, v in example.items():
                    if v is not None:
                        example_gpu[k] = v.to(self.device)

                with torch.cuda.amp.autocast(
                    enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    # example_gpu['lr_pair_flag']=torch.rand([1, 343, 11, 4]).to(self.device)
                    # print(example_gpu)
                    loss = model(**example_gpu) / grad_accum
                # Backward; accumulate gradients and clip by grad norm
                loss.backward()
                loss_during_accum.append(loss.item())
                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if self.config['max_grad_norm']:
                        norm_plm = torch.nn.utils.clip_grad_norm_(
                            plm_param,
                            self.config['max_grad_norm'],
                            error_if_nonfinite=False
                        )
                        norm_task = torch.nn.utils.clip_grad_norm_(
                            task_param,
                            self.config['max_grad_norm'],
                            error_if_nonfinite=False
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    if type(self.optimizer) == FusedAdam:
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.zero_grad(set_to_none=True)

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if self.scheduler._step_count % self.config['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / self.config['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info(
                            'Step %d: avg loss %.2f; steps/sec %.2f' %
                            (self.scheduler._step_count, avg_loss,
                            self.config['report_frequency'] / (end_time - start_time))
                        )
                        start_time = end_time
                        wandb.log({'avg_loss': avg_loss}) 

                    # Evaluate
                    if self.scheduler._step_count % self.config['eval_frequency'] == 0:
                        logger.info('Dev')

                        f1, _ = self.evaluate(
                            model, examples_dev, stored_info, self.scheduler._step_count
                        )
                        logger.info('Test')
                        f1_test = 0.
                        if f1 > max_f1:
                            max_f1 = max(max_f1, f1)
                            max_f1_test = 0. 
                            self.save_model_checkpoint(
                                model, self.optimizer, self.scheduler, self.scheduler._step_count, epo
                            )

                        logger.info('Eval max f1: %.2f' % max_f1)
                        logger.info('Test max f1: %.2f' % max_f1_test)
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % self.scheduler._step_count)

        return

    def evaluate(
        self, model, tensor_examples, stored_info, step, predict=False
    ):
        # use different evaluator for different task
        # should return two values: f1, metrics
        # f1 is used for model selection, the higher the better
        raise NotImplementedError()


    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        plm_param, task_param = model.get_params(named=True)

        grouped_param = [
            {
                'params': [p for n, p in plm_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['plm_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in plm_param if any(nd in n for nd in no_decay)],
                'lr': self.config['plm_learning_rate'],
                'weight_decay': 0.0
            }, {
                'params': [p for n, p in task_param],
                'lr': self.config['task_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        if self.config["optimizer"].lower() == 'adamw':
            # FusedAdam is faster. Requires apex.
            # Otherwise use AdamW
            opt_class = FusedAdam

        logger.info(opt_class)
        optimizer = opt_class(
            grouped_param,
            lr=self.config['plm_learning_rate'],
            eps=self.config['adam_eps']
        )
        return optimizer

    def get_scheduler(self, optimizer, total_update_steps):
        # Only warm up plm lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        lr_lambda_plm = util.get_scheduler_lambda(
            self.config['plm_scheduler'], warmup_steps, total_update_steps)
        lr_lambda_task = util.get_scheduler_lambda(
            self.config['task_scheduler'], 0, total_update_steps)

        scheduler = LambdaLR(optimizer, [
            lr_lambda_plm, # parameters with decay
            lr_lambda_plm, # parameters without decay
            lr_lambda_task
        ])
        return scheduler

    def save_model_checkpoint(self, model, optimizer, scheduler, step, current_epoch):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save({
            'current_epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, path_ckpt)
        logger.info('Saved model, optmizer, scheduler to %s' % path_ckpt)
        return

    def load_model_checkpoint(self, model, suffix, continue_training=True):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        checkpoint = torch.load(path_ckpt, map_location=torch.device('cpu'))

        if type(checkpoint) is dict:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if continue_training:
                self.optimizer = self.get_optimizer(model)

                epochs, grad_accum = self.config['num_epochs'], self.config['gradient_accumulation_steps']
                batch_size = self.config['batch_size']
                total_update_steps = len(self.data.get_tensor_examples()[0]) *\
                                      epochs // (grad_accum * batch_size)
                self.scheduler = self.get_scheduler(self.optimizer, total_update_steps)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                current_epoch = checkpoint['current_epoch']

                logger.info('Loaded model, optmizer, scheduler from %s' % path_ckpt)
                return model, current_epoch
            else:
                return model, -1
        else:
            model.load_state_dict(checkpoint, strict=False)
            logger.info('Loaded model from %s' % path_ckpt)
            return model, -1
