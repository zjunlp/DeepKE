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
import bmtrain as bmt
from .layernorm import LayerNorm
from .attention import Attention
from .feedforward import FeedForward


class SelfAttentionBlock(bmt.DistributedModule):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dtype=torch.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
    ):

        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.self_attention = Attention(
            dim_model=dim_model,
            num_heads=num_heads,
            dim_head=dim_head,
            dtype=dtype,
            dropout_p=dropout_p,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation.
            position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of attention block.

        """  # noqa: E501
        x = self.layernorm_before_attention(hidden_states)
        x = self.self_attention(x, x, attention_mask, position_bias, use_cache, past_key_value)
        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None

        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = (hidden_states + x) / 1.05

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


class FFNBlock(torch.nn.Module):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dtype=torch.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.ffn = FeedForward(
            dim_model,
            dim_ff,
            dtype=dtype,
            dropout_p=dropout_p,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """  # noqa: E501
        x = self.layernorm_before_ffn(hidden_states)
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = (hidden_states + x) / 1.05
        return hidden_states


class TransformerBlock(torch.nn.Module):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        dtype=torch.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        mask_att: bool = False,
        mask_ffn: bool = False,
    ):
        super().__init__()
        self.mask_att = mask_att
        self.mask_ffn = mask_ffn

        if not self.mask_att:
            self.self_att = SelfAttentionBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_head=dim_head,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
            )

        if not self.mask_ffn:
            self.ffn = FFNBlock(
                dim_model=dim_model,
                dim_ff=dim_ff,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
            )

    def forward(
        self,
        self_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        self_position_bias: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            self_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation of self-attention.
            self_position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of transformer block.

        """  # noqa: E501
        # (batch, dim_model, seq_self)
        current_key_value = None
        if not self.mask_att:
            hidden_states = self.self_att(
                self_hidden_states,
                attention_mask=self_attention_mask,
                position_bias=self_position_bias,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
            if use_cache:
                hidden_states, current_key_value = hidden_states
        else:
            hidden_states = self_hidden_states

        # (batch, dim_model, seq_self)
        if not self.mask_ffn:
            hidden_states = self.ffn(hidden_states)

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states
