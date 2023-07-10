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

from typing import Optional, Tuple
import torch
import math
from .linear import Linear


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.half,
        dropout_p: Optional[float] = None,
    ) -> None:

        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.project_q = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_k = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_v = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)

        self.attention_out = Linear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype)

        self.softmax = torch.nn.Softmax(dim=-1)

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            hidden_q (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        h_q = self.project_q(hidden_q)
        h_k = self.project_k(hidden_kv)
        h_v = self.project_v(hidden_kv)

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        h_k = h_k.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        h_v = h_v.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        if past_kv is not None:
            h_k = torch.cat([past_kv[0], h_k], dim=-2)
            h_v = torch.cat([past_kv[1], h_v], dim=-2)
            len_k = h_k.size(-2)

        # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
        score = torch.matmul(h_q, h_k.transpose(-1, -2)) / math.sqrt(self.dim_head)
        score = score + position_bias
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )

        score = self.softmax(score)

        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )

        if self.dropout is not None:
            score = self.dropout(score)

        # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
        score = score.contiguous().view(batch_size, len_q, self.num_heads * self.dim_head)

        score = self.attention_out(score)
        if use_cache:
            return score, (h_k, h_v)
        else:
            return score
