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

from typing import List, Optional, Tuple
from typing_extensions import TypedDict
import torch

from ..utils import Config
from ..layers import Encoder, EmbeddingExt, BucketPositionBias
import bmtrain as bmt
from ..utils.gradient_shrink import gradient_shrink


class CPMBeeInferenceState(TypedDict):
    buffer_position: torch.Tensor
    buffer_context: torch.Tensor
    buffer_sample_ids: torch.Tensor
    buffer_num_segments: torch.Tensor
    buffer_segments: torch.Tensor
    buffer: List[Tuple[torch.Tensor, torch.Tensor]]


class CPMBeeConfig(Config):
    def __init__(
        self,
        vocab_size=30720,
        dim_model=4096,
        num_heads=64,
        dim_head=64,
        dim_ff=10240,
        num_layers=32,
        dropout_p=0.0,
        position_bias_num_buckets=256,
        position_bias_num_segment_buckets=256,
        position_bias_max_distance=2048,
        eps=1e-6,
        half: bool = True,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):

        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_num_segment_buckets = position_bias_num_segment_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.vocab_size = vocab_size
        self.mask_modules = mask_modules


class CPMBee(bmt.DistributedModule):
    def __init__(self, config: CPMBeeConfig):

        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            mask_modules=config.mask_modules,
        )

        self.input_embedding = EmbeddingExt(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            init_std=0.02,
        )

        self.position_bias = BucketPositionBias(
            num_heads=config.num_heads,
            num_buckets=config.position_bias_num_buckets,
            num_segment_bucket=config.position_bias_num_segment_buckets,
            max_distance=config.position_bias_max_distance,
            dtype=config.dtype,
        )

    def forward(
        self,
        input: torch.Tensor,  # (batch, seqlen) int32
        input_sub: torch.Tensor,  # (batch, seqlen) int32
        length: torch.Tensor,  # (batch) int32
        context: torch.Tensor,  # (batch, seqlen) bool
        sample_ids: torch.Tensor,  # (batch, seq_len) int32
        num_segments: torch.Tensor,  # (batch, seq_len) int32
        segment: torch.Tensor,  # (batch, seqlen) int32
        segment_rel_offset: torch.Tensor,  # (batch, seq_len) int32
        segment_rel: torch.Tensor,  # (batch, num_segment_bucket) int32
        span: torch.Tensor,  # (batch, seqlen) int32
        ext_table_ids: torch.Tensor,  # (ext_table_size) int32
        ext_table_sub: torch.Tensor,  # (ext_table_size) int32
    ):
        batch = input.size(0)
        seqlen = input.size(1)
        # processing masks and position bias bucket
        with torch.no_grad():
            device = input.device

            # calc segment bucket
            segment_rel_2d = torch.masked_fill(
                segment[:, :, None] * num_segments[:, :, None]
                + segment[:, None, :]
                + segment_rel_offset[:, :, None],
                ~(
                    (sample_ids[:, :, None] == sample_ids[:, None, :])
                    & (span[:, None, :] == span[:, :, None])
                ),  # not in the same span or sample
                0,  # avoid torch.gather overflow
            ).view(batch, seqlen * seqlen)

            segment_bucket = torch.gather(
                input=segment_rel,
                dim=1,
                index=segment_rel_2d.long(),
            ).view(batch, seqlen, seqlen)

            segment_bucket.masked_fill_(
                ~(
                    (sample_ids[:, :, None] == sample_ids[:, None, :])
                    & (span[:, None, :] == span[:, :, None])
                ),  # not in the same span or sample
                1,  # bucket is used for in-context samples
            )

            # directional mask
            directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(
                seqlen, device=device
            ).view(-1, 1)
            # sample mask
            sample_mask_2d = (sample_ids[:, :, None] == 0) | (
                sample_ids[:, :, None] == sample_ids[:, None, :]
            )
            # context mask
            attention_mask = context[:, None, :] | (
                context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
            )
            # span mask
            attention_mask = (
                attention_mask & sample_mask_2d & (span[:, None, :] == span[:, :, None])
            )
            # length mask
            mask_1d = (
                torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
            )
            attention_mask = (
                mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
            )
            position = torch.arange(seqlen, device=device).expand(batch, seqlen)

        hidden_states = self.input_embedding(input, input_sub)
        position_bias = self.position_bias(position, position, segment_bucket)

        hidden_states = self.encoder(hidden_states, attention_mask, position_bias)

        ext_table = self.input_embedding(ext_table_ids, ext_table_sub)

        logits = self.input_embedding.projection(hidden_states, ext_table)
        return logits, hidden_states

    def inference(
        self,
        input: torch.Tensor,  # (batch, len_q) int32
        input_sub: torch.Tensor,  # (batch, len_q) int32
        position: torch.Tensor,  # (batch, len_q)  int32
        context: torch.Tensor,  # (batch, len_q) bool
        sample_ids: torch.Tensor,  # (batch, len_q) int32
        num_segments: torch.Tensor,  # (batch, len_q) int32
        segment: torch.Tensor,  # (batch, len_q) int32
        segment_rel_offset: torch.Tensor,  # (batch, len_q) int32
        segment_rel: torch.Tensor,  # (batch, num_segment_bucket) int32
        ext_table_ids: torch.Tensor,  # (ext_table_size) int32
        ext_table_sub: torch.Tensor,  # (ext_table_size) int32
        past_key_values: Optional[CPMBeeInferenceState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, CPMBeeInferenceState]:
        with torch.no_grad():
            if past_key_values is None:
                present_position = position
                present_context = context
                present_sample_ids = sample_ids
                present_num_segments = num_segments
                present_segments = segment
                present_buffer = None
            else:
                present_position = torch.cat([past_key_values["buffer_position"], position], dim=-1)
                present_context = torch.cat([past_key_values["buffer_context"], context], dim=-1)
                present_sample_ids = torch.cat(
                    [past_key_values["buffer_sample_ids"], sample_ids], dim=-1
                )
                present_num_segments = torch.cat(
                    [past_key_values["buffer_num_segments"], num_segments], dim=-1
                )
                present_segments = torch.cat([past_key_values["buffer_segments"], segment], dim=-1)
                present_buffer = past_key_values["buffer"]

            batch = input.size(0)
            len_q = input.size(1)
            len_buffer = present_position.size(1)

            segment_rel_2d = torch.masked_fill(
                segment[:, :, None] * num_segments[:, :, None]
                + present_segments[:, None, :]
                + segment_rel_offset[:, :, None],
                ~(
                    (sample_ids[:, :, None] == present_sample_ids[:, None, :])
                ),  # not in the same sample
                0,  # avoid torch.gather overflow
            ).view(batch, len_q * len_buffer)

            segment_bucket = torch.gather(
                input=segment_rel,
                dim=1,
                index=segment_rel_2d.long(),
            ).view(batch, len_q, len_buffer)

            segment_bucket.masked_fill_(
                ~(
                    (sample_ids[:, :, None] == present_sample_ids[:, None, :])
                ),  # not in the same span or sample
                1,  # bucket is used for in-context samples
            )

            # directional mask
            directional_mask_2d = present_position[:, None, :] <= position[:, :, None]
            # sample mask
            sample_mask_2d = (sample_ids[:, :, None] == 0) | (
                sample_ids[:, :, None] == present_sample_ids[:, None, :]
            )
            # context mask
            attention_mask = present_context[:, None, :] | (
                context[:, :, None].logical_not()
                & directional_mask_2d.view(batch, len_q, len_buffer)
            )
            # span mask
            attention_mask = attention_mask & sample_mask_2d
            # length mask
            mask_1d = present_num_segments != 0
            attention_mask = mask_1d.view(batch, 1, len_buffer) & attention_mask

            hidden_states = gradient_shrink(self.input_embedding(input, input_sub))

            position_bias = gradient_shrink(
                self.position_bias(position, present_position, segment_bucket)
            )
            hidden_states, present_key_values = self.encoder(
                hidden_states,
                attention_mask,
                position_bias,
                True,
                present_buffer,
            )
            ext_table = gradient_shrink(self.input_embedding(ext_table_ids, ext_table_sub))
            logits = self.input_embedding.projection(hidden_states, ext_table)

            return (
                logits,
                hidden_states,
                {
                    "buffer_position": present_position,
                    "buffer_context": present_context,
                    "buffer_sample_ids": present_sample_ids,
                    "buffer_num_segments": present_num_segments,
                    "buffer_segments": present_segments,
                    "buffer": present_key_values,
                },
            )
