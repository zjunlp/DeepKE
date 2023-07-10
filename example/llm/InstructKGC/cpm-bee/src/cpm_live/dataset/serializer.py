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

import pickle
import json


class Serializer:
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        raise NotImplementedError()

    def deserialize(self, data: bytes):
        raise NotImplementedError()


class PickleSerializer(Serializer):
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, data: bytes):
        return pickle.loads(data)


class JsonSerializer(Serializer):
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

    def deserialize(self, data: bytes):
        return json.loads(data.decode("utf-8"))


class RawSerializer(Serializer):
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        return obj

    def deserialize(self, data: bytes):
        return data
