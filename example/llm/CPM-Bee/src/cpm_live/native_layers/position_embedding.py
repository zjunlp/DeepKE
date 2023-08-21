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
import math
from typing import Union
import torch
import torch.nn.functional as F


class SegmentPositionEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        num_segments=1,
        num_buckets=32,
        max_distance=128,
        bidirectional=False,
        dtype=torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):

        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_segments = num_segments

        self.relative_attention_bias = torch.nn.parameter.Parameter(
            torch.empty(num_segments * num_segments + num_buckets, num_heads, dtype=dtype)
        )

    def forward(
        self,
        key_pos: torch.Tensor,
        query_pos: torch.Tensor,
        key_segment: torch.Tensor,
        query_segment: torch.Tensor,
    ):
        with torch.no_grad():

            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            assert key_pos.size(0) == query_pos.size(0)
            assert keylen == key_segment.size(1) and querylen == query_segment.size(1)

            key_pos = key_pos.view(batch, -1, keylen)
            query_pos = query_pos.view(batch, querylen, -1)
            key_segment = key_segment.view(batch, -1, keylen)
            query_segment = query_segment.view(batch, querylen, -1)

            relative_position_bucket = self._segment_relative_position_bucket(
                query_segment, key_segment
            )
            relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠

            # b*q*k
            absolute_position_bucket = self._position_bucket(
                torch.arange(keylen, dtype=torch.int32, device=relative_position_bucket.device)[
                    None, :
                ]
                - torch.arange(querylen, dtype=torch.int32, device=relative_position_bucket.device)[
                    :, None
                ],
                bidirectional=self.bidirectional,
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            relative_position_bucket = torch.where(
                (key_segment == query_segment),
                absolute_position_bucket[None, :, :],
                relative_position_bucket,
            )
            # (batch, len_q, len_k)

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment

    def _position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position.to(torch.int32), relative_postion_if_large
        )
        return relative_buckets


class BucketPositionBias(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        num_segment_bucket: int = 32,
        max_distance: int = 128,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.num_segment_bucket = num_segment_bucket
        self.max_distance = max_distance

        self.relative_attention_bias = torch.nn.parameter.Parameter(
            torch.empty(num_buckets + num_segment_bucket, num_heads, dtype=dtype)
        )

    def forward(
        self,
        query_pos: torch.Tensor,  # (batch, len_q)
        key_pos: torch.Tensor,  # (batch, len_k)
        rel_buckets: torch.Tensor,  # (batch, len_q, len_k)
    ):
        with torch.no_grad():

            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            assert key_pos.size(0) == query_pos.size(0)
            assert (
                rel_buckets.size(0) == batch
                and rel_buckets.size(1) == querylen
                and rel_buckets.size(2) == keylen
            )

            relative_position_bucket = rel_buckets - 1 + self.num_buckets  # 与相对位置编码区间不重叠

            # b*q*k
            inner_segment_bucket = self._position_bucket(
                key_pos[..., None, :] - query_pos[..., :, None],
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            relative_position_bucket = torch.where(
                rel_buckets == 0,
                inner_segment_bucket,
                relative_position_bucket,
            )
            # (batch, len_q, len_k)

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position.to(torch.int32), relative_postion_if_large
        )
        return relative_buckets


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        base=10000,
        distance_scale: Union[int, float] = 1,
        dtype: torch.dtype = torch.half,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim)
        )
        inv_freq = inv_freq.to(dtype)
        self.distance_scale = distance_scale
        self.dtype = dtype
        self.inv_freq = inv_freq

    def forward(self, x: torch.Tensor, x_pos: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(..., dim)``): Inputs.
            x_pos (:obj:`torch.Tensor` of shape ``(...)``): Positions of inputs.
        """
        inv_freq = self.inv_freq.to(device=x.device, dtype=x.dtype)

        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None] * inv_freq[None, :]  # (..., dim/2)

        # the same implementation as sat
        emb = torch.cat((freqs, freqs), dim=-1)  # (..., dim)
        emb_cos = emb.cos()  # (..., dim)
        emb_sin = emb.sin()  # (..., dim)

        rotate_x = torch.cat(
            [-x[..., x.size(-1) // 2 :], x[..., : x.size(-1) // 2]], dim=-1
        )  # (..., dim)

        return x * emb_cos + rotate_x * emb_sin
