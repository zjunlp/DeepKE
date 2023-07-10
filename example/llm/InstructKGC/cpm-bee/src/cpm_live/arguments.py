# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse


def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group("model", "model configuration")
    group.add_argument("--model-config", type=str, help="model configuration file")
    return parser


def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group("train", "training configurations")

    group.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset")
    group.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a directory containing a model checkpoint.",
    )
    group.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )
    group.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Output filename to save checkpoints to.",
    )

    group.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="tensorboard directory",
    )

    group.add_argument("--inspect-iters", type=int, default=1000, help="number of inspecting")
    group.add_argument("--batch-size", type=int, default=32, help="Data Loader batch size")
    group.add_argument("--clip-grad", type=float, default=1.0, help="gradient clipping")
    group.add_argument(
        "--train-iters",
        type=int,
        default=1000000,
        help="total number of iterations to train over all training runs",
    )
    group.add_argument("--max-length", type=int, default=512, help="max length of input")

    group.add_argument("--seed", type=int, default=1234, help="random seed for reproducibility")

    # Learning rate.
    group.add_argument("--lr", type=float, default=1.0e-4, help="initial learning rate")
    group.add_argument("--weight-decay", type=float, default=1.0e-2, help="weight decay rate")
    group.add_argument("--loss-scale", type=float, default=65536, help="loss scale")

    group.add_argument(
        "--warmup-iters",
        type=int,
        default=100,
        help="steps for learning rate warm-up",
    )
    group.add_argument(
        "--lr-decay-style",
        type=str,
        default="noam",
        choices=["constant", "linear", "cosine", "exponential", "noam"],
        help="learning rate decay function",
    )
    group.add_argument("--lr-decay-iters", type=int, default=None, help="lr decay steps")
    group.add_argument(
        "--start-step", type=int, default=0, help="step to start or continue training"
    )

    return parser


def add_pretrain_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("pretrain", "pretrain configurations")
    group.add_argument(
        "--save-iters",
        type=int,
        default=1000,
        help="number of iterations between saves",
    )
    group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="log directory",
    )

    return parser


def add_finetune_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("finetune", "fintune configurations")
    group.add_argument("--epoch", type=int, default=1, help="number of training epochs")
    group.add_argument("--task-name", type=str, default="task", help="name of training task")
    group.add_argument(
        "--use-delta",
        action="store_true",
        default=False,
        help="use delta tuning or not"
    )
    group.add_argument("--eval_dataset", type=str, help="path to eval dataset")
    group.add_argument(
        "--drop-last",
        action="store_true",
        default=False,
        help="drop data from each epoch that cannot be formed into a complete batch at the end",
    )
    group.add_argument("--eval-interval", type=int, default=500, help="eval interval")
    group.add_argument("--early-stop-patience", type=int, default=5, help="early stop steps")
    return parser


def get_args(pretrain: bool = False, finetune: bool = False):
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    if pretrain:
        parser = add_pretrain_args(parser)
    if finetune:
        parser = add_finetune_args(parser)

    args = parser.parse_args()
    return args
