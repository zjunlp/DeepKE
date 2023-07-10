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

import json
import time
from typing import Any, Dict, List, Union
import torch
import bmtrain as bmt
import os
from cpm_live.arguments import get_args

from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from cpm_live.utils import allgather_objects, LogManager
from cpm_live.training_tasks.bee import MixedDataset


def get_tokenizer(args):
    tokenizer = CPMBeeTokenizer()
    return tokenizer


def get_model(args):
    config = CPMBeeConfig.from_json_file(args.model_config)
    model = CPMBee(config)
    if args.load is not None:
        bmt.load(model, args.load)
    else:
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay
    )
    if args.load is not None:
        if os.path.exists(os.path.join(args.save, args.save_name + (".rank-%d.opt" % 0))):
            # optimizer state exists
            states = torch.load(
                os.path.join(args.save, args.save_name + (".rank-%d.opt" % bmt.rank()))
            )
            optimizer.load_state_dict(states)
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
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2333)
    args = get_args(pretrain=True)
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


class LossSpikeDetector:
    def __init__(self, log_path: str) -> None:
        self._last_loss: Dict[str, float] = {}
        self._last_data: List[Any] = [None]
        self._log_path = log_path

    def update_data(self, data: Any):
        self._last_data.append(data)
        if len(self._last_data) > 2:
            self._last_data = self._last_data[-2:]

    def update_loss(self, iteration: int, loss_map: Dict[str, float]):
        loss_spike_result = []
        for task, loss in loss_map.items():
            if task in self._last_loss:
                if loss > self._last_loss[task] * 3:
                    # loss spike!
                    loss_spike_result.append(
                        {
                            "prev": self._last_loss[task],
                            "curr": loss,
                            "task": task,
                        }
                    )
            self._last_loss[task] = float(loss)
        if len(loss_spike_result) > 0:
            self._write_log(iteration, self._last_data[-1], loss_spike_result)

    def _write_log(self, iteration: int, data: Any, result: List[Dict[str, Any]]):
        with open(self._log_path, "a", encoding="utf-8") as fp:
            fp.write("=" * 20)
            fp.write("\nloss spike at {}\n".format(iteration))
            fp.write("{}\n".format(json.dumps(result, indent=4, ensure_ascii=False)))
            fp.write("data: \n")
            for d in data:
                fp.write("{}\n".format(json.dumps(d, indent=4, ensure_ascii=False)))
            fp.write("\n\n")


def pretrain(
    args,
    tokenizer: CPMBeeTokenizer,
    model: CPMBee,
    optimizer: bmt.optim.AdamOffloadOptimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
    optim_manager: bmt.optim.OptimManager,
):

    average_time = bmt.utils.AverageRecorder()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step

    lsd = LossSpikeDetector("debug/spile.%d.log" % bmt.rank())

    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    if args.log_dir is not None and bmt.rank() == 0:
        log_mgr = LogManager(args.log_dir)

    global_token_pass = 0.0
    global_world_size = bmt.world_size()
    dataloader = MixedDataset(
        args.dataset, args.batch_size, args.max_length, tokenizer, max_depth=8
    )

    if os.path.exists(os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))):
        # load dataset states if exists
        dataset_states = torch.load(
            os.path.join(args.save, args.save_name + ("-%d.data.pt" % start_step))
        )
        missing = dataloader.load_state_dict(dataset_states)
        if len(missing) > 0:
            bmt.print_rank("Missing keys when loading dataset states: ", missing)
    dataloader.start()
    try:
        for iteration, data in enumerate(dataloader):

            iteration = iteration + start_step + 1
            assert data["inputs"].shape[0] == args.batch_size
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
            lsd.update_data(data["raw_data"])

            # ===========
            optim_manager.zero_grad()
            # torch.cuda.empty_cache()
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
            global_loss = bmt.sum_loss(loss).item()
            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            # ===========
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            # ===========
            current_stream = torch.cuda.current_stream()
            # some reduce ops of distributed parameter were launched on load stream
            current_stream.wait_stream(bmt.config['load_stream'])
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
                for i in range(task_num):
                    task_loss = loss_func(
                        logits.view(-1, logits.size(-1)), targets_tmp[i, :].view(-1)
                    )
                    # global_task_loss = float(bmt.sum_loss(task_loss).item())
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
            lsd.update_loss(iteration, task_loss_map)

            train_info = {
                "time": tim_usage["init"],
                "iteration": iteration,
                "loss": global_loss,
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
                    "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} |"
                    + " token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}"
                ).format(
                    iteration,
                    global_loss,
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
            if args.log_dir is not None and bmt.rank() == 0:
                log_mgr.write(**train_info)
            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
                writer.add_scalar("Optimizer/lr", lr_scheduler.current_lr, iteration)
                writer.add_scalar("Optimizer/scale", optim_manager.loss_scale, iteration)
                writer.add_scalar("Optimizer/grad_norm", grad_norm.item(), iteration)
                for task_name, loss in task_loss_map.items():
                    writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)

            if args.save is not None and iteration % args.save_iters == 0:
                bmt.save(model, os.path.join(args.save, args.save_name + ("-%d.pt" % iteration)))
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(args.save, args.save_name + (".rank-%d.opt" % bmt.rank())),
                )
                all_states = dataloader.state_dict()
                if bmt.rank() == 0:
                    # rank 0 writes the dataloader state
                    torch.save(
                        all_states,
                        os.path.join(args.save, args.save_name + ("-%d.data.pt" % iteration)),
                    )
                del all_states
    finally:
        dataloader.close()

    bmt.save(model, os.path.join(args.save, args.save_name + ".pt"))


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler, optim_manager = setup_model_and_optimizer(args)
    pretrain(args, tokenizer, model, optimizer, lr_scheduler, optim_manager)


if __name__ == "__main__":
    main()
