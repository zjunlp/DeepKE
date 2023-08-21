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

import io
import os
import struct
from typing import List, Optional, Set
import torch
import bisect
import bmtrain as bmt
import json
from .serializer import Serializer, PickleSerializer
import random
import string
import time


def _random_string():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


_DEFAULT_BLOCK_SIZE = 16 << 20


class FileInfo:
    def __init__(
        self,
        file_name: str = "",
        block_begin: int = 0,
        block_end: int = 0,
        nbytes: int = 0,
        nlines: int = 0,
        mask: bool = False,
        block_size: int = _DEFAULT_BLOCK_SIZE,
    ) -> None:
        self.file_name = file_name
        self.block_begin = block_begin
        self.block_end = block_end
        self.nbytes = nbytes
        self.nlines = nlines
        self.mask = mask
        self.block_size = block_size

    def state_dict(self):
        return {
            "file_name": self.file_name,
            "block_begin": self.block_begin,
            "block_end": self.block_end,
            "nbytes": self.nbytes,
            "nlines": self.nlines,
            "mask": self.mask,
            "block_size": self.block_size,
        }

    def load_state_dict(self, d):
        self.file_name = d["file_name"]
        self.block_begin = d["block_begin"]
        self.block_end = d["block_end"]
        self.nbytes = d["nbytes"]
        self.nlines = d["nlines"]
        self.mask = d["mask"]
        self.block_size = d["block_size"]

    def dumps(self) -> str:
        return json.dumps(self.state_dict())

    def loads(self, data: str) -> "FileInfo":
        self.load_state_dict(json.loads(data))
        return self

    def dump(self, fp: io.TextIOWrapper) -> "FileInfo":
        fp.write(self.dumps())
        return self

    def load(self, fp: io.TextIOWrapper) -> "FileInfo":
        self.loads(fp.read())
        return self


def _read_info_list(meta_path: str) -> List[FileInfo]:
    info: List[FileInfo] = []
    while True:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) > 0:
                        info.append(FileInfo().loads(line))
            return info
        except Exception as e:
            print("Error: reading info list in _read_info_list!,meta_path={path}, err={err}".
                  format(path=meta_path, err=str(e)))
            time.sleep(10)


def _write_info_list(meta_path: str, info: List[FileInfo]):
    base_path = os.path.dirname(meta_path)
    random_fname = os.path.join(base_path, ".meta.bin.%s" % _random_string())
    while True:
        try:
            with open(random_fname, "w", encoding="utf-8") as f:
                for v in info:
                    f.write(v.dumps() + "\n")
            os.rename(random_fname, meta_path)
            return
        except Exception:
            print("Error: writing info list!")
            time.sleep(10)


def _filtered_range(
    begin: int, end: int, rank: int, world_size: int, filter_set: Optional[Set[int]] = None
):
    begin = begin + (rank + (world_size - (begin % world_size))) % world_size

    if filter_set is not None:
        return [i for i in range(begin, end, world_size) if i in filter_set]
    else:
        return [i for i in range(begin, end, world_size)]


# for some bugs that may exist in hdfs
class SafeFile:

    def __init__(self, fname, mode):
        self.fname = None
        self.mode = None
        self._fp = None
        self.open_file(fname, mode)

    def read(self, size=-1):
        if self._fp is None:
            raise RuntimeError("Dataset is closed")
        try:
            res = self._fp.read(size)
            self.offset = self._fp.tell()
            return res
        except Exception as e:
            print("Error {}: reading blocks in read {}!".format(e, self.fname))
            self.open_file(self.fname, self.mode, self.offset)
            return self.read(size)

    def tell(self):
        if self._fp is None:
            raise RuntimeError("Dataset is closed")
        try:
            res = self._fp.tell()
            self.offset = res
            return res
        except Exception as e:
            print("Error {}: reading blocks in tell {}!".format(e, self.fname))
            self.open_file(self.fname, self.mode, self.offset)
            return self.tell()

    def seek(self, offset, whence=0):
        if self._fp is None:
            raise RuntimeError("Dataset is closed")
        try:
            res = self._fp.seek(offset, whence)
            self.offset = self._fp.tell()
            return res
        except Exception as e:
            print("Error {}: reading blocks in seek {}!".format(e, self.fname))
            self.open_file(self.fname, self.mode, self.offset)
            return self.seek(offset, whence)

    def close(self):
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
        self._fp = None

    def open_file(self, fname, mode, offset=None):
        if not os.path.exists(fname):
            raise RuntimeError("Dataset does not exist")
        try:
            self.fname = fname
            self.mode = mode
            self._fp = open(fname, mode)
            if offset is not None:
                self._fp.seek(offset, io.SEEK_SET)
            self.offset = self._fp.tell()
        except Exception as e:
            print("Error {}: reading blocks in open_file {}!".format(e, self.fname))
            time.sleep(10)
            self.open_file(fname, mode, offset)


class DistributedDataset:
    """Open dataset in readonly mode.

    `DistributeDataset` is used to read datasets in a distributed manner.
    Data in this dataset will be distributed evenly in blocks to each worker in the `distributed communicator`.

    **Note** When all data has been read, reading dataset again will revert back to the first data.

    Args:
        path (str): Path to dataset.
        rank (int): Rank in distributed communicator. See: bmtrain.rank()
        world_size (int): Total workers in distributed communicator. See: bmtrain.world_size()
        block_size (int): Size of each block in bytes. All files in the same dataset should have the same block size. Default: 16MB

    Example:
        >>> dataset = DistributedDataset("/path/to/dataset")
        >>> for i in range(10):
        >>>     dataset.read()
    """  # noqa: E501

    def __init__(
        self,
        path: str,
        rank: int = 0,
        world_size: int = 1,
        serializer: Optional[Serializer] = None,
        max_repeat_times: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        # config
        self._path = path
        self._rank = rank
        self._world_size = world_size
        self._max_repeat_times = max_repeat_times
        self._repeat_times = 0
        self._shuffle = shuffle

        if serializer is None:
            serializer = PickleSerializer()
        self.serializer = serializer

        # dataset meta
        self._unused_block: List[int] = []
        self._file_info: List[FileInfo] = []
        self._file_ends: List[int] = []
        self._total_blocks = 0
        self._nbytes = 0
        self._nlines = 0

        # states
        self._curr_block = None
        self._fp = None

        # cache
        self._last_mod_time = 0
        self._curr_fname = None

        self._update_states(fast_skip=False)
        self._repeat_times += 1

    def _update_states(self, fast_skip: bool = True):
        meta_path = os.path.join(self._path, "meta.bin")

        while True:
            try:
                mod_time = os.stat(meta_path).st_mtime
                break
            except Exception as e:
                print("Error: reading info list in DistributedDataset._update_states, "
                      "meta_path={path}, err={err}!".format(path=meta_path, err=str(e)))
                time.sleep(10)

        if self._last_mod_time < mod_time:
            # file changed
            pass
        else:
            if fast_skip:
                return

        info: List[FileInfo] = []
        if os.path.exists(meta_path):
            info = _read_info_list(meta_path)

        old_len = len(self._file_info)
        if old_len > len(info):
            raise RuntimeError("Dataset meta file: changed unexpectly")

        mask_changed = False
        for i in range(old_len):
            if self._file_info[i].file_name != info[i].file_name:
                raise RuntimeError("Dataset meta file: changed unexpectly")
            if self._file_info[i].block_begin != info[i].block_begin:
                raise RuntimeError("Dataset meta file: changed unexpectly")
            if self._file_info[i].block_end != info[i].block_end:
                raise RuntimeError("Dataset meta file: changed unexpectly")
            if self._file_info[i].mask != info[i].mask:
                mask_changed = True

        if info[0].block_begin != 0:
            raise RuntimeError("Dataset meta file: block error (0)")
        for i in range(len(info) - 1):
            if info[i].block_end != info[i + 1].block_begin:
                raise RuntimeError("Dataset meta file: block error (%d)" % (i + 1))

        if (old_len == len(info) and not mask_changed) and fast_skip:
            # fast skip
            return

        if len(info) > 0:
            total_blocks = info[-1].block_end
            self._nbytes = 0
            self._nlines = 0
            for v in info:
                self._nbytes += v.nbytes
                self._nlines += v.nlines
        else:
            total_blocks = 0
            self._nbytes = 0
            self._nlines = 0

        if total_blocks > 0:
            unused_block_set = set(self._unused_block)
            nw_unused_block: List[int] = []
            for i in range(len(info)):
                v = info[i]
                if not v.mask:
                    if i < old_len:
                        nw_unused_block.extend(
                            _filtered_range(
                                v.block_begin,
                                v.block_end,
                                self._rank,
                                self._world_size,
                                unused_block_set,
                            )
                        )
                    else:
                        nw_unused_block.extend(
                            _filtered_range(
                                v.block_begin, v.block_end, self._rank, self._world_size
                            )
                        )

            # re-shuffle unused blocks
            if self._shuffle:
                random.shuffle(nw_unused_block)
            self._unused_block = nw_unused_block

            self._file_ends = []
            for v in info:
                self._file_ends.append(v.block_end)
        else:
            self._unused_block = []
            self._file_ends = []
        self._total_blocks = total_blocks
        self._file_info = info

        assert len(self._file_ends) == len(self._file_info)

    def _mask_file(self, f: FileInfo):
        self._unused_block = [
            block_id
            for block_id in self._unused_block
            if block_id < f.block_begin or block_id >= f.block_end
        ]

    def _get_block_file(self, block_id: int):
        # find block in which file
        file_idx = bisect.bisect_right(self._file_ends, block_id)
        return self._file_info[file_idx]

    def _prepare_new_epoch(self):
        if self._max_repeat_times is not None:
            if self._repeat_times >= self._max_repeat_times:
                raise EOFError("End of dataset")
        nw_unused_block: List[int] = []
        for v in self._file_info:
            if not v.mask:
                nw_unused_block.extend(
                    _filtered_range(v.block_begin, v.block_end, self._rank, self._world_size)
                )
        if self._shuffle:
            random.shuffle(nw_unused_block)
        self._unused_block = nw_unused_block
        self._repeat_times += 1

    def _get_next_block(self):
        self._update_states()
        if len(self._unused_block) == 0:
            self._prepare_new_epoch()
            if len(self._unused_block) == 0:
                raise RuntimeError("Empty dataset {}".format(self._path))

        mn_block: int = self._unused_block.pop()
        return mn_block

    def _state_dict(self):
        self._update_states()
        num_unused_block = len(self._unused_block)
        if (self._fp is not None) and (self._curr_block is not None):
            curr_block = self._curr_block
            curr_f = self._get_block_file(curr_block)
            inblock_offset = self._fp.tell() - (curr_block - curr_f.block_begin) * curr_f.block_size
        else:
            curr_block = -1
            inblock_offset = 0

        return {
            "states": torch.tensor(self._unused_block, dtype=torch.long, device="cpu"),
            "block": torch.tensor(
                [curr_block, inblock_offset, num_unused_block, self._repeat_times],
                dtype=torch.long,
                device="cpu",
            ),
        }

    def state_dict(self):
        """Returns a state dict representing the read states of the dataset.

        Example:
            >>> state = dataset.state_dict()
            >>> dataset.load_state_dict(state)
        """
        self._update_states()
        num_unused_block = len(self._unused_block)

        if (self._fp is not None) and (self._curr_block is not None):
            curr_block = self._curr_block
            curr_f = self._get_block_file(curr_block)
            inblock_offset = self._fp.tell() - (curr_block - curr_f.block_begin) * curr_f.block_size
        else:
            curr_block = -1
            inblock_offset = 0

        with torch.no_grad():
            if self._world_size > 1:
                gpu_num_unused_block = torch.tensor([num_unused_block], dtype=torch.long).cuda()
                max_unused_blocks = (
                    bmt.distributed.all_reduce(gpu_num_unused_block, op="max").cpu().item()
                )
                gpu_states = torch.full((max_unused_blocks,), -1, dtype=torch.long).cuda()
                gpu_states[:num_unused_block] = torch.tensor(
                    self._unused_block, dtype=torch.long
                ).cuda()

                gpu_block = torch.tensor(
                    [curr_block, inblock_offset, num_unused_block, self._repeat_times],
                    dtype=torch.long,
                ).cuda()
                global_states = bmt.distributed.all_gather(
                    gpu_states
                ).cpu()  # (world_size, max_unused_blocks)
                global_block = bmt.distributed.all_gather(gpu_block).cpu()  # (world_size, 4)
                return {"states": global_states, "block": global_block}
            else:
                return {
                    "states": torch.tensor([self._unused_block], dtype=torch.long, device="cpu"),
                    "block": torch.tensor(
                        [[curr_block, inblock_offset, num_unused_block, self._repeat_times]],
                        dtype=torch.long,
                        device="cpu",
                    ),
                }

    def load_state_dict(self, state, strict: bool = True):
        """Load dataset state.

        Args:
            state (dict): dataset state dict.
            strict (bool): If `strict` is True, world size needs to be the same as when exported.

        Example:
            >>> state = dataset.state_dict()
            >>>
        """
        block_states: torch.LongTensor = state["states"]
        block_info: torch.LongTensor = state["block"]

        if block_states.size(0) != self._world_size:
            if strict:
                raise ValueError(
                    "world_size changed (%d -> %d)" % (state["block"].size(0), self._world_size)
                )
            else:
                self._curr_block = None
                self._fp = None
                self._curr_fname = None
                self._repeat_times = int(block_info[0, 3].item())

                # re-shuffle unused blocks
                nw_unused_block: List[int] = []
                for i in range(block_states.size(0)):
                    # filter blocks that are not in this rank
                    num_unused_blocks: int = int(block_info[i, 2].item())
                    nw_unused_block.extend(
                        [
                            block_id
                            for block_id in block_states[i, :num_unused_blocks].tolist()
                            if block_id % self._world_size == self._rank
                        ]
                    )
                if self._shuffle:
                    random.shuffle(nw_unused_block)
                self._unused_block = nw_unused_block
        else:
            curr_block, inblock_offset, num_unused_blocks, self._repeat_times = block_info[
                self._rank
            ].tolist()

            if curr_block == -1:
                self._curr_block = None
            else:
                while True:
                    try:
                        self._curr_block = curr_block
                        f_info = self._get_block_file(self._curr_block)
                        self._open_file(
                            f_info.file_name,
                            (self._curr_block - f_info.block_begin)
                            * f_info.block_size
                            + inblock_offset,
                        )
                        self._unused_block = block_states[self._rank, :num_unused_blocks].tolist()
                        break
                    except Exception:
                        print("Error: reading block!")
                        time.sleep(10)
        # end
        self._update_states()

    def _get_file_path(self, fname):
        return os.path.join(self._path, fname)

    def _open_file(self, fname, offset):
        if self._curr_fname != fname:
            if self._fp is not None:
                self._fp.close()
                self._curr_fname = None
            # self._fp = open(self._get_file_path(fname), "rb")
            self._fp = SafeFile(self._get_file_path(fname), "rb")
            self._curr_fname = fname
        else:
            assert self._fp is not None, "Unexpected error"
        self._fp.seek(offset, io.SEEK_SET)  # move to block

    def read(self):
        """Read a piece of data from dataset.

        Workers in different ranks will read different data.
        """
        if self._curr_block is None:
            next_block_id = self._get_next_block()
            f_info = self._get_block_file(next_block_id)
            try:
                self._open_file(
                    f_info.file_name,
                    (next_block_id - f_info.block_begin) * f_info.block_size,
                )
                self._curr_block = next_block_id
            except FileNotFoundError:
                print("ERR: reading again!")
                self._mask_file(f_info)
                return self.read()  # read again

        if self._fp is None:
            raise RuntimeError("Dataset is not initialized")

        MAGIC = self._fp.read(1)
        if MAGIC == b"\x1F":
            # correct
            size = struct.unpack("I", self._fp.read(4))[0]
            data = self._fp.read(size)
            return self.serializer.deserialize(data)
        elif MAGIC == b"\x00":
            # end of block
            self._curr_block = None
            return self.read()  # read next block
        else:
            raise ValueError("Invalid magic header")

    @property
    def nbytes(self):
        return self._nbytes


class SimpleDataset(DistributedDataset):
    def __init__(
        self,
        path: str,
        serializer: Optional[Serializer] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__(
            path,
            0,
            1,
            serializer=serializer,
            max_repeat_times=1,
            shuffle=shuffle,
        )

    def __iter__(self):
        while True:
            try:
                data = self.read()
            except EOFError:
                self._repeat_times = 0
                break
            yield data

    def __len__(self):
        return self._nlines


class DatasetWriter:
    def __init__(self, fname: str, block_size: int, serializer: Optional[Serializer] = None):
        self._fname = fname
        self._block_size = block_size
        self._fp = open(self._fname, "wb")
        self._inblock_offset = 0

        self._nbytes = 0
        self._nlines = 0
        self._nblocks = 1

        if serializer is None:
            serializer = PickleSerializer()
        self.serializer = serializer

    def write(self, data):
        """Write a piece of data into dataset.

        Args:
            data (Any): Serialization will be done using pickle.

        Example:
            >>> writer.write( "anything you want" )

        """
        byte_data = self.serializer.serialize(data)
        byte_data = struct.pack("I", len(byte_data)) + byte_data
        if self._inblock_offset + 2 + len(byte_data) > self._block_size:
            self._fp.write(
                b"\x00" * (self._block_size - self._inblock_offset)
            )  # fill the remaining space with 0
            self._inblock_offset = 0
            self._nblocks += 1
            # we go to the next block
        if self._inblock_offset + 2 + len(byte_data) > self._block_size:
            raise ValueError("data is larger than block size")

        self._nbytes += len(byte_data)
        self._nlines += 1

        self._inblock_offset += 1 + len(byte_data)
        self._fp.write(b"\x1F")
        self._fp.write(byte_data)

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def nblocks(self):
        return self._nblocks

    @property
    def nlines(self):
        return self._nlines

    def close(self):
        if not self._fp.closed:
            self._fp.write(b"\x00" * (self._block_size - self._inblock_offset))
            self._fp.close()


class DatasetBuilder:
    def __init__(
        self,
        path: str,
        dbname: str,
        block_size=_DEFAULT_BLOCK_SIZE,
        serializer: Optional[Serializer] = None,
    ) -> None:
        self._block_size = block_size
        self._path = path
        self._dbname = dbname
        if serializer is None:
            serializer = PickleSerializer()
        self.serializer = serializer

        if not os.path.exists(self._path):
            os.makedirs(self._path)

        meta_path = os.path.join(self._path, "meta.bin")

        info: List[FileInfo] = []
        if os.path.exists(meta_path):
            info = _read_info_list(meta_path)

        for v in info:
            if v.file_name == dbname:
                raise ValueError("Dataset name exists")

        self._db_path = os.path.join(self._path, self._dbname)
        if os.path.exists(self._db_path):
            raise ValueError("File exists `%s`" % self._db_path)

    def __enter__(self):
        self._writer = DatasetWriter(self._db_path, self._block_size, self.serializer)
        return self._writer

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._writer is None:
            raise RuntimeError("Unexpected call to __exit__")

        self._writer.close()
        if exc_type is not None:
            print("Error while writing file")
            if os.path.exists(self._db_path):
                os.unlink(self._db_path)
        else:
            meta_path = os.path.join(self._path, "meta.bin")
            info: List[FileInfo] = []
            if os.path.exists(meta_path):
                info = _read_info_list(meta_path)
            last_block = 0
            if len(info) > 0:
                last_block = info[-1].block_end
            info.append(
                FileInfo(
                    self._dbname,
                    last_block,
                    last_block + self._writer.nblocks,
                    self._writer.nbytes,
                    self._writer.nlines,
                    False,
                    self._block_size,
                )
            )

            # atomic write to meta file
            _write_info_list(meta_path, info)

        self._writer = None


def build_dataset(
    path: str,
    dbname: str,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    serializer: Optional[Serializer] = None,
):
    """Open the dataset in write mode and returns a writer.

    Args:
        path (str): Path to dataset.
        dbname (str): The name of the file to which the data will be written. The `dbname` needs to be unique in this `dataset`.
        block_size (int): Size of each block in bytes. All files in the same dataset should have the same block size. Default: 16MB

    Example:
        >>> with build_dataset("/path/to/dataset", "data_part_1") as writer:
        >>>     for i in range(10):
        >>>         writer.write( { "anything you want" } )
    """  # noqa: E501
    return DatasetBuilder(path, dbname, block_size=block_size, serializer=serializer)
