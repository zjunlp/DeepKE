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

from typing import Optional
import torch
import bmtrain as bmt
from .linear import Linear


class DenseGatedACT(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        dtype=torch.half,
    ):
        super().__init__()

        self.w_0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale_before=False,
        )

        self.w_1 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale_before=False,
        )
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor):
        """This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)

        """  # noqa: E501
        gate_score = self.act(self.w_0(x))
        x = self.w_1(x)

        x = gate_score * x
        return x


class FeedForward(bmt.DistributedModule):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dtype=torch.half,
        dropout_p: Optional[float] = None,
    ):

        super().__init__()

        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
            dtype=dtype,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.w_out = Linear(
            dim_in=dim_ff,
            dim_out=dim_model,
            dtype=dtype,
            scale_before=False,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """  # noqa: E501
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)

        return x
