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

import os
import struct
from typing import List, Optional
from .distributed_dataset import (
    SimpleDataset,
    build_dataset,
    _read_info_list,
    _write_info_list,
    _random_string,
    _DEFAULT_BLOCK_SIZE,
    FileInfo,
)
from .serializer import RawSerializer
import random
import shutil

try:
    from tqdm import tqdm

    support_tqdm = True
except ModuleNotFoundError:
    support_tqdm = False


_DEFAULT_SHUFFLE_BUCKET_SIZE = 1 << 30


def shuffle_dataset(
    path_src: str,
    path_tgt: str,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    bucket_size: int = _DEFAULT_SHUFFLE_BUCKET_SIZE,
    progress_bar: bool = False,
    output_name: Optional[str] = None,
):
    """Shuffle one distributed datataset, write results to another dataset.

    Args:
        path_str (str): path to source dataset
        path_tgt (str): path to write results
        block_size (int): dataset block size (default: 16MB)
        bucket_size (int): shuffle algorithm bucket size (default: 1GB)
        progress_bar (bool): show progress bar

    Example:
        >>> shuffle_dataset("/path/to/source", "/path/to/output")
    """

    if progress_bar and not support_tqdm:
        raise RuntimeError("Requires `tqdm` to enable progress bar.")

    ds = SimpleDataset(path_src, serializer=RawSerializer())
    num_buckets = (ds.nbytes + bucket_size - 1) // bucket_size

    tmp_files = [os.path.join(path_src, ".tmp.%s" % _random_string()) for _ in range(num_buckets)]

    try:
        # Step 1: write to bucket randomly
        f_tmp = [open(fname, "wb") for fname in tmp_files]
        try:
            iterator = ds
            if progress_bar:
                iterator = tqdm(ds, desc="Shuffle step 1/2")
            for data in iterator:
                bucket_id = int(random.random() * num_buckets)
                len_data = len(data)
                f_tmp[bucket_id].write(struct.pack("I", len_data) + data)
        finally:
            # close all files
            for fp in f_tmp:
                if not fp.closed:
                    fp.close()
        f_tmp = []

        # Step 2: shuffle inside bucket
        if output_name is None:
            output_name = "%s.shuffle" % _random_string()
        with build_dataset(
            path_tgt,
            output_name,
            block_size=block_size,
            serializer=RawSerializer(),
        ) as writer:
            iterator = tmp_files
            if progress_bar:
                iterator = tqdm(tmp_files, desc="Shuffle step 2/2")

            for fname in iterator:
                fp = open(fname, "rb")
                data_in_bucket = []
                while True:
                    try:
                        raw_data = fp.read(4)
                        if len(raw_data) == 0:
                            # EOF
                            break
                        len_data = struct.unpack("I", raw_data)[0]
                        data_in_bucket.append(fp.read(len_data))
                    except EOFError:
                        break
                random.shuffle(data_in_bucket)
                for data in data_in_bucket:
                    writer.write(data)
                fp.close()
                os.unlink(fname)
    finally:
        # cleanup
        for fname in tmp_files:
            if os.path.exists(fname):
                os.unlink(fname)


def compact_dataset(path: str):
    """Compact the dataset, removes blocks which the files were deleted.

    **Note** This may affect the existing dataset state dict.

    Args:
        path (str): path to dataset

    Example:
        >>> compact_dataset("/path/to/dataset")

    """

    meta_path = os.path.join(path, "meta.bin")

    info: List[FileInfo] = []
    if os.path.exists(meta_path):
        info = _read_info_list(meta_path)
    else:
        raise ValueError("Dataset not exists")

    nw_info: List[FileInfo] = []
    curr_block = 0
    for v in info:
        if not os.path.exists(v.file_name):
            # file is deleted
            pass
        else:
            num_file_block = v.block_end - v.block_begin
            nw_info.append(
                FileInfo(
                    v.file_name,
                    curr_block,
                    curr_block + num_file_block,
                    v.nbytes,
                    v.nlines,
                    v.mask,
                    v.block_size,
                )
            )
            curr_block += num_file_block

    _write_info_list(meta_path, nw_info)


def mask_dataset(path: str, dbname: str, mask: bool = True):
    """Mask one file in dataset. Blocks in masked datasets won't be read later.

    Args:
        path (str): path to dataset
        dbname (str): file name in this dataset which you want to mask
        mask (bool): True for mask, False for unmask

    Example:
        >>> mask_dataset("/path/to/dataset", "data_part_1", mask=True)

    """

    meta_path = os.path.join(path, "meta.bin")

    info: List[FileInfo] = []
    if os.path.exists(meta_path):
        info = _read_info_list(meta_path)
    else:
        raise ValueError("Dataset not exists")

    for v in info:
        if v.file_name == dbname:
            v.mask = mask
    _write_info_list(meta_path, info)


def merge_dataset(dst: str, src: str):

    meta_path_src = os.path.join(src, "meta.bin")
    meta_path_dst = os.path.join(dst, "meta.bin")

    info_src: List[FileInfo] = []
    if os.path.exists(meta_path_src):
        info_src = _read_info_list(meta_path_src)
    else:
        raise ValueError("Dataset not exists")

    info_dst: List[FileInfo] = []
    if os.path.exists(meta_path_dst):
        info_dst = _read_info_list(meta_path_dst)
    else:
        raise ValueError("Dataset not exists")

    curr_block = 0
    nw_info: List[FileInfo] = []
    for v in info_dst:
        num_file_block = v.block_end - v.block_begin
        nw_info.append(
            FileInfo(
                v.file_name,
                curr_block,
                curr_block + num_file_block,
                v.nbytes,
                v.nlines,
                v.mask,
                v.block_size,
            )
        )
        curr_block += num_file_block

    for v in info_src:
        num_file_block = v.block_end - v.block_begin

        dst_db_name = os.path.join(dst, v.file_name)
        nw_fname = v.file_name
        if os.path.exists(dst_db_name):
            idx = 0
            while os.path.exists(dst_db_name + "_{}".format(idx)):
                idx += 1
            dst_db_name = dst_db_name + "_{}".format(idx)
            nw_fname = nw_fname + "_{}".format(idx)

        shutil.copy(os.path.join(src, v.file_name), dst_db_name)
        nw_info.append(
            FileInfo(
                nw_fname,
                curr_block,
                curr_block + num_file_block,
                v.nbytes,
                v.nlines,
                v.mask,
                v.block_size,
            )
        )
        curr_block += num_file_block

    _write_info_list(meta_path_dst, nw_info)
