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
import os
import copy
from typing import Any, Dict, Union
from .log import logger


def load_dataset_config(dataset_path: str):
    cfg = json.load(open(dataset_path, "r", encoding="utf-8"))

    platform_config_path = os.getenv("PLATFORM_CONFIG_PATH")
    if platform_config_path is None:
        logger.info(
            "no platform_config_path. Directly load dataset_path({dataset_path})"
            .format(dataset_path=dataset_path)
        )
        return cfg

    path_dict = json.load(open(platform_config_path, "r", encoding="utf-8"))["dataset_map"]
    logger.info(
        "load dataset_path({dataset_path}) with platform_config_path({platform_config_path})"
        .format(dataset_path=dataset_path, platform_config_path=platform_config_path)
    )
    for dataset in cfg:
        dataset["path"] = os.path.join(path_dict[dataset["dataset_name"]], dataset["path"])
        dataset["transforms"] = os.path.join(
            path_dict[dataset["dataset_name"]], dataset["transforms"]
        )
    return cfg


class Config(object):
    """model configuration"""

    def __init__(self):
        super().__init__()

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **args):
        config_dict = cls._dict_from_json_file(json_file, **args)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike], **args):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        res = json.loads(text)
        for key in args:
            res[key] = args[key]
        return res

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output
