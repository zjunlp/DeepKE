from deepke.triple_extraction.ASP import *
import sys
import logging
import random
import numpy as np

import torch

from torch.optim import AdamW

import deepke.triple_extraction.ASP.util as util

from deepke.triple_extraction.ASP.util.multigpu_fused_adam import FusedAdam

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import time
from os.path import join
from datetime import datetime

from deepke.triple_extraction.ASP.util.runner import Runner

from metrics import EREEvaluator

import wandb
import hydra


class ERERunner(Runner):

    def evaluate(self, model, tensor_examples, stored_info, step, predict=False):
        evaluator = EREEvaluator()

        eval_batch_size = 32
        if "pp" in self.name or "11b" in self.name or "xxl" in self.name:
            eval_batch_size = 24
        if "doclevel" in self.name:
            eval_batch_size = 4
        util.runner.logger.info('Step %d: evaluating on %d samples with batch_size %d' % (
            step, len(tensor_examples), eval_batch_size))

        evalloader = DataLoader(
            tensor_examples, batch_size=eval_batch_size, shuffle=False, 
            num_workers=0,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        model.eval()
        for i, (doc_keys, tensor_example) in enumerate(evalloader):
            example_gpu = {}

            for k, v in tensor_example.items():
                if v is not None:
                    example_gpu[k] = v.to(self.device)
            example_gpu['is_training'][:] = 0

            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=self.use_amp, dtype=torch.bfloat16
            ):
                output = model(**example_gpu)

            for batch_id, doc_key in enumerate(doc_keys):

                gold_res = model.extract_gold_res_from_gold_annotation(
                    {k:v[batch_id] for k, v in tensor_example.items()}, 
                    stored_info['example'][doc_key]
                )
                decoded_results = model.decoding(
                    {k:v[batch_id] for k,v in output.items()}, 
                    stored_info['example'][doc_key]
                )

                decoded_results.update(
                    gold_res
                )
                evaluator.update(
                    **decoded_results
                )
                if predict:
                    util.runner.logger.info(stored_info['example'][doc_key])
                    util.runner.logger.info(decoded_results)

        p,r,f = evaluator.get_prf()
        metrics = {
            'Eval_Ent_Precision': p[0] * 100,
            'Eval_Ent_Recall': r[0] * 100,
            'Eval_Ent_F1': f[0] * 100,
            'Eval_Rel_Precision': p[1] * 100,
            'Eval_Rel_Recall': r[1] * 100,
            'Eval_Rel_F1': f[1] * 100,
            'Eval_Rel_p_Precision': p[2] * 100,
            'Eval_Rel_p_Recall': r[2] * 100,
            'Eval_Rel_p_F1': f[2] * 100,
        }
        for k,v in metrics.items():
            util.runner.logger.info('%s: %.4f'%(k, v))

        return f[1] * 100, metrics
        
wandb.init(project="DeepKE_TRIPLE_ASP")
wandb.watch_called = False
def main():
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    saved_suffix = sys.argv[3] if len(sys.argv) >= 4 else None
    runner = ERERunner(
        config_file="conf/train.yaml",
        config_name=config_name,
        gpu_id=gpu_id
    )

    if saved_suffix is not None:
        model, start_epoch = runner.initialize_model(saved_suffix, continue_training=True)
        runner.train(model, continued=True, start_epoch=start_epoch)
    else:
        model, _ = runner.initialize_model()
        runner.train(model, continued=False)

# python run_ere.py t5_base 0
if __name__ == '__main__':
    main()

