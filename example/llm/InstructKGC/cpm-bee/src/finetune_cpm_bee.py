# coding=utf-8
# Copyright 2022 The OpenBMB team.
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

import time
from typing import Dict, List, Union
import torch
import bmtrain as bmt
import os
from opendelta import LoraModel
from cpm_live.arguments import get_args

from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from cpm_live.utils import allgather_objects
from cpm_live.training_tasks.bee import FinetuneDataset


def get_tokenizer(args):
    tokenizer = CPMBeeTokenizer()
    return tokenizer


def get_model(args):
    config = CPMBeeConfig.from_json_file(args.model_config)
    model = CPMBee(config)
    model.config = config
    if args.load is not None:
        bmt.load(model, args.load)
    else:
        bmt.init_parameters(model)
    # insert LoRA
    if args.use_delta:
        delta_model = LoraModel(
            backbone_model=model, modified_modules=["project_q", "project_v"], backend="bmt"
        )
        delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
        delta_model.log()
    return model


def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay
    )
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_scheduler = bmt.lr_scheduler.Noam(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler


def setup_model_and_optimizer(args):
    model = get_model(args)
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    optim_manager = bmt.optim.OptimManager(
        loss_scale=args.loss_scale,
        loss_scale_factor=2,
        loss_scale_steps=512,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    return tokenizer, model, optimizer, lr_scheduler, optim_manager


def initialize():
    args = get_args(finetune=True)
    bmt.init_distributed(seed=args.seed)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage


def evaluation(model, args, tokenizer, loss_func):
    bmt.print_rank("evaluation begins...")
    eval_dataloader = FinetuneDataset(
        args.eval_dataset,
        1,
        args.max_length,
        tokenizer,
        max_depth=8,
        task_name=args.task_name,
        drop_last=args.drop_last,
    )
    eval_losses = []
    last_data = None
    with torch.no_grad():
        for iteration, data in enumerate(eval_dataloader):
            iteration = iteration + 1
            skip_this_batch = False
            if data is None:
                if last_data is None:
                    raise RuntimeError(
                        "Dataset is too small, please use a smaller batch size or sequence length!"
                    )
                data = last_data
                skip_this_batch = True
            else:
                last_data = data

            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_ids_sub = torch.from_numpy(data["inputs_sub"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            input_context = torch.from_numpy(data["context"]).cuda().bool()
            input_sample_ids = torch.from_numpy(data["sample_ids"]).cuda().to(torch.int32)
            input_num_segments = torch.from_numpy(data["num_segments"]).cuda().to(torch.int32)
            input_segment_ids = torch.from_numpy(data["segment_ids"]).cuda().to(torch.int32)
            input_segment_rel_offset = (
                torch.from_numpy(data["segment_rel_offset"]).cuda().to(torch.int32)
            )
            input_segment_rel = torch.from_numpy(data["segment_rel"]).cuda().to(torch.int32)
            input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            ext_table_ids = torch.from_numpy(data["ext_ids"]).cuda().to(torch.int32)
            ext_table_sub = torch.from_numpy(data["ext_sub"]).cuda().to(torch.int32)
            # ===========
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # ===========
            logits, _ = model(
                input_ids,
                input_ids_sub,
                input_length,
                input_context,
                input_sample_ids,
                input_num_segments,
                input_segment_ids,
                input_segment_rel_offset,
                input_segment_rel,
                input_span,
                ext_table_ids,
                ext_table_sub,
            )

            loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
            if skip_this_batch:
                loss = loss * 0
            eval_losses.append(bmt.sum_loss(loss))

        overall_loss = torch.stack(eval_losses).mean().item()
    return overall_loss


def finetune(
    args,
    tokenizer: CPMBeeTokenizer,
    model: CPMBee,
    optimizer: bmt.optim.AdamOffloadOptimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
    optim_manager: bmt.optim.OptimManager,
):

    average_time = bmt.utils.AverageRecorder()
    if model.config.dtype == torch.half:
        loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    else:
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    best_eval_loss, eval_loss_increase = 1e9, 0
    global_token_pass = 0.0
    global_steps = 0
    global_world_size = bmt.world_size()
    dataloader = FinetuneDataset(
        args.dataset,
        args.batch_size,
        args.max_length,
        tokenizer,
        max_depth=8,
        task_name=args.task_name,
        drop_last=args.drop_last,
    )

    for epoch in range(args.epoch):
        epoch = epoch + 1
        last_data = None
        for iteration, data in enumerate(dataloader):
            iteration = iteration + 1
            global_steps = global_steps + 1
            skip_this_batch = False
            if data is None:
                if last_data is None:
                    raise RuntimeError(
                        "Dataset is too small, please use a smaller batch size or sequence length!"
                    )
                data = last_data  # use last data
                skip_this_batch = True
            else:
                last_data = data

            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_ids_sub = torch.from_numpy(data["inputs_sub"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            input_context = torch.from_numpy(data["context"]).cuda().bool()
            input_sample_ids = torch.from_numpy(data["sample_ids"]).cuda().to(torch.int32)
            input_num_segments = torch.from_numpy(data["num_segments"]).cuda().to(torch.int32)
            input_segment_ids = torch.from_numpy(data["segment_ids"]).cuda().to(torch.int32)
            input_segment_rel_offset = (
                torch.from_numpy(data["segment_rel_offset"]).cuda().to(torch.int32)
            )
            input_segment_rel = torch.from_numpy(data["segment_rel"]).cuda().to(torch.int32)
            input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            ext_table_ids = torch.from_numpy(data["ext_ids"]).cuda().to(torch.int32)
            ext_table_sub = torch.from_numpy(data["ext_sub"]).cuda().to(torch.int32)
            task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            task_names = data["task_names"]
            # ===========
            optim_manager.zero_grad()
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # ===========
            logits, _ = model(
                input_ids,
                input_ids_sub,
                input_length,
                input_context,
                input_sample_ids,
                input_num_segments,
                input_segment_ids,
                input_segment_rel_offset,
                input_segment_rel,
                input_span,
                ext_table_ids,
                ext_table_sub,
            )
            loss = loss_func(logits.view(-1, logits.size(-1)), targets.long().view(-1))
            if skip_this_batch:
                loss = loss * 0

            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            # ===========
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            # ===========
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=1.0)
            optim_manager.step()
            mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)

            # ==========
            iteration_time = tim_usage["optim"] - tim_usage["init"]
            average_time.record(iteration_time)

            with torch.no_grad():
                task_num = len(task_names)
                targets_tmp = targets.expand(task_num, -1, -1)
                task = torch.arange(task_num, dtype=torch.int32, device="cuda")[:, None, None]
                targets_tmp = torch.where(
                    task_ids == task,
                    targets_tmp,
                    torch.scalar_tensor(-100, dtype=torch.int32, device="cuda"),
                )

                task_loss_map: Dict[str, float] = {}
                if not skip_this_batch:
                    for i in range(task_num):
                        task_loss = loss_func(
                            logits.view(-1, logits.size(-1)), targets_tmp[i, :].long().view(-1)
                        )
                        task_loss_map[task_names[i]] = task_loss.item()
                gatherd_task_loss_map: List[Dict[str, float]] = allgather_objects(task_loss_map)

                global_task_loss_map: Dict[str, Union[List[float], float]] = {}
                for local_task_loss_map in gatherd_task_loss_map:
                    for task_name, task_loss in local_task_loss_map.items():
                        if task_name not in global_task_loss_map:
                            global_task_loss_map[task_name] = []
                        global_task_loss_map[task_name].append(task_loss)

                task_loss_map = {}
                for task_name in sorted(list(global_task_loss_map.keys())):
                    avg_loss = sum(global_task_loss_map[task_name]) / len(
                        global_task_loss_map[task_name]
                    )
                    task_loss_map[task_name] = avg_loss

            local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
            local_total_rate = bmt.sum_loss(local_total_rate).item()
            global_token_pass += (
                global_world_size * local_total_rate * args.max_length * args.batch_size
            )
            avg_time = average_time.value

            train_info = {
                "time": tim_usage["init"],
                "epoch": epoch,
                "iteration": iteration,
                "loss": task_loss_map[args.task_name],
                "lr": lr_scheduler.current_lr,
                "lr_scale": int(optim_manager.loss_scale),
                "time_usage": tim_usage,
                "mem_usage": mem_usage,
                "avg_time": avg_time,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / avg_time,
                "grad_norm": grad_norm.item(),
                "mask_max": ((targets >= 0).sum(-1).float().mean() / args.max_length).item(),
                "num_gpus": global_world_size,
                "task_loss": task_loss_map,
            }

            bmt.print_rank(
                (
                    "| Epoch: {:3d} | Iter: {:6d} | loss: {:.4f} "
                    + "| lr: {:.4e}, scale: {:10.4f} | time: {:.4f} |"
                    + " token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}"
                ).format(
                    epoch,
                    iteration,
                    task_loss_map[args.task_name],
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    avg_time,
                    input_length.float().mean() / args.max_length,
                    (targets >= 0).sum(-1).float().mean() / args.max_length,
                    grad_norm,
                )
            )
            bmt.print_rank(
                "| "
                + " | ".join(
                    [
                        "{} loss: {:.4f}".format(task_name, loss)
                        for task_name, loss in task_loss_map.items()
                    ]
                )
            )
            if iteration % args.inspect_iters == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                bmt.print_rank(bmt.inspect.format_summary(model_inspect))
                train_info["model_inspect"] = model_inspect

            # write log here
            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", task_loss_map[args.task_name], global_steps)
                for task_name, loss in task_loss_map.items():
                    writer.add_scalar("Loss/train/{}".format(task_name), loss, global_steps)

            # evaluation
            if global_steps % args.eval_interval == 0:
                eval_loss = evaluation(model, args, tokenizer, loss_func)
                if args.tensorboard is not None and bmt.rank() == 0:
                    writer.add_scalar("Loss/eval", eval_loss, global_steps)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    eval_loss_increase = 0
                    if args.save is not None:
                        if not args.use_delta:
                            bmt.save(model, os.path.join(args.save, args.save_name + "-best.pt"))
                        else:
                            state_dict = model.state_dict()
                            if bmt.rank() == 0:
                                torch.save(state_dict, os.path.join(args.save, args.save_name + "-delta-best.pt"))
                else:
                    eval_loss_increase += 1
                bmt.print_rank(
                    "| Eval loss: {:.4f} | Increase: {:2d}".format(eval_loss, eval_loss_increase)
                )
                if eval_loss_increase == args.early_stop_patience:
                    bmt.print_rank(
                        "Eval loss has increased {:d} times, the finetune loop early stopped."
                        .format(eval_loss_increase)
                    )
                    return
    # end of finetune


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler, optim_manager = setup_model_and_optimizer(args)
    finetune(args, tokenizer, model, optimizer, lr_scheduler, optim_manager)


if __name__ == "__main__":
    main()
