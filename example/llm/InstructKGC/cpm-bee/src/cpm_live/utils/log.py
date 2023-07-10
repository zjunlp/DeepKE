import os
import sys
from typing import Any, Dict, Optional, Tuple, Union
import datetime
import json
import logging
import bmtrain as bmt


# Set up the common logger
def _get_logger():
    log = logging.getLogger('__name__')
    log.setLevel(logging.INFO)
    console_handle = logging.StreamHandler(sys.stdout)
    node_name = os.getenv("NODE_NAME", str(bmt.rank()))
    console_handle.setFormatter(
        logging.Formatter(
            '[%(levelname)s][%(asctime)s][{}][%(filename)s:%(lineno)d:%(process)d] - %(message)s'
            .format(node_name),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    log.addHandler(console_handle)
    return log


logger = _get_logger()


class LogManager:
    def __init__(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        now = self.get_log_time()
        latest_log: Union[Dict[str, Any], None] = None
        for _ in range(15):
            log_name = self.get_log_name(now)
            if os.path.exists(log_name):
                with open(log_name, "r") as flog:
                    latest_log = json.loads(flog.readlines()[-1])  # get last log
                break
            now -= datetime.timedelta(days=1)

        if latest_log is None:
            self.global_token_pass = 0
        else:
            self.global_token_pass = latest_log["token pass"]

    def get_log_time(self) -> datetime.datetime:
        return datetime.datetime.utcnow() + datetime.timedelta(hours=16)

    def get_log_name(self, now: Optional[datetime.datetime] = None):
        if now is None:
            now = self.get_log_time()
        return os.path.join(self.path, "log.%s.txt" % now.strftime("%Y%m%d"))

    def write(
        self,
        time: float,
        iteration: int,
        loss: float,
        lr: float,
        lr_scale: float,
        time_usage: Dict[str, float],
        mem_usage: Dict[str, Tuple[float, float]],
        avg_time: float,
        token_max: float,
        token_pass: float,
        throughout: float,
        grad_norm: float,
        mask_max: float,
        num_gpus: int,
        task_loss: Dict[str, float],
        model_inspect: Optional[Any] = None,
    ):
        with open(self.get_log_name(), "a") as fp:
            ret = {
                "time": time,
                "iter": iteration,
                "loss": loss,
                "lr": lr,
                "lr scale": int(lr_scale),
                "time usage": time_usage,
                "mem usage": mem_usage,
                "avg time (s)": avg_time,
                "token/max": token_max,
                "token pass": token_pass + self.global_token_pass,
                "throughout (token/s)": throughout,
                "grad_norm": grad_norm,
                "mask/max": mask_max,
                "num_gpus": num_gpus,
                "task_loss": task_loss,
            }
            if model_inspect is not None:
                ret["model_inspect"] = model_inspect
            fp.write(json.dumps(ret, ensure_ascii=False) + "\n")
