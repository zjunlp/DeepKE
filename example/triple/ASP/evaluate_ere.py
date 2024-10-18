from deepke.triple_extraction.ASP import *
from run_ere import ERERunner
import sys
import torch
import wandb
wandb.init(project="DeepKE_TRIPLE_ASP")
wandb.watch_called = False

def evaluate(config_name, gpu_id, saved_suffix):
    runner = ERERunner(
        config_file="conf/train.yaml",
        config_name=config_name,
        gpu_id=gpu_id
    )
    model, _ = runner.initialize_model(saved_suffix, continue_training=False)

    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    runner.evaluate(model, examples_test, stored_info, 0, predict=True)  # Eval test

# E.g.
# CUDA_VISIBLE_DEVICES=0 python evaluate_ere.py CMeIE Mar05_19-39-56_2000 0
if __name__ == '__main__':
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    evaluate(config_name, gpu_id, saved_suffix)
