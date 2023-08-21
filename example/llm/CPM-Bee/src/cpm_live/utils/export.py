import os
import time
import functools
import torch
import bmtrain as bmt
import json
from cpm_live.models import CPMBee
from .log import logger
from typing import List, Optional


def rename_if_exists(file_path):
    if not os.path.exists(file_path):
        return
    timestamp = time.strftime('%Y%m%d%H%M%S')
    file_dir, file_name = os.path.split(file_path)
    file_root, file_ext = os.path.splitext(file_name)
    new_file_name = f"{file_root}_bak_{timestamp}{file_ext}"
    new_file_path = os.path.join(file_dir, new_file_name)
    try:
        os.rename(file_path, new_file_path)
        logger.info(f"File '{file_name}' already exists. Renamed to '{new_file_name}'")
    except Exception as e:
        logger.warn(
            "rename file failed,file_path={file_path}, new_file_path={new_file_path},err={err}"
            .format(file_path=file_path, new_file_path=new_file_path, err=str(e)))


def rename_if_exists_decorator(func):
    @functools.wraps(func)
    def wrapper(file_path, *args, **kwargs):
        rename_if_exists(file_path)
        return func(file_path, *args, **kwargs)
    return wrapper


@rename_if_exists_decorator
def bmt_save(file_path: str, model: CPMBee, export_files: Optional[List[str]] = None):
    bmt.save(model, file_path)
    if export_files is not None:
        export_files.append(file_path)


@rename_if_exists_decorator
def torch_save(file_path: str, obj: object, export_files: Optional[List[str]] = None):
    torch.save(obj, file_path)
    if export_files is not None:
        export_files.append(file_path)


@rename_if_exists_decorator
def json_save(file_path: str, obj: object, export_files: Optional[List[str]] = None):
    with open(file_path, "w") as data_f:
        json.dump(obj, data_f)
    if export_files is not None:
        export_files.append(file_path)
