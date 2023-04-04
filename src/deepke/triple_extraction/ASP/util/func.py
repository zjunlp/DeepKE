import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import math
import logging
import random
from typing import Optional, Tuple, Any, Dict, Iterable, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN

logger = logging.getLogger(__name__)

def flatten(l):
    return [item for sublist in l for item in sublist]

def initialize_config(config_name, config_file="experiments.conf"):
    logger.info("Running experiment: {}".format(config_name))

    config = pyhocon.ConfigFactory.parse_file(join("./", config_file))[config_name]
    config['log_dir'] = join(config["log_root"], config_name)
    makedirs(config['log_dir'], exist_ok=True)

    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    makedirs(config['tb_dir'], exist_ok=True)

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def bucket_distance(offsets):
    """ offsets: [num spans1, num spans2] """
    # 10 semi-logscale bin: 0, 1, 2, 3, 4, (5-7)->5, (8-15)->6, (16-31)->7, (32-63)->8, (64+)->9
    logspace_distance = torch.log2(offsets.to(torch.float)).to(torch.long) + 3
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on
    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer
    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        # we want the input relative_position to be negative in this case
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def batch_select(tensor, idx, device=None):
    """ Do selection per row (first axis). """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]
    
    if device is None:
        device = tensor.device

    tensor = tensor.reshape(dim0_size * dim1_size, -1)
    idx_offset = (torch.arange(0, dim0_size, device=device) * dim1_size)
    for _ in range(len(idx.size()) - 1):
        idx_offset = idx_offset.unsqueeze(-1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]

    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        selected = selected.squeeze(-1)

    return selected


def batch_add(tensor, idx, val, device=None):
    """ Do addition per row (first axis). """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]
    
    if device is None:
        device = tensor.device

    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = (torch.arange(0, dim0_size, device=device) * dim1_size)
    for _ in range(len(idx.size()) - 1):
        idx_offset = idx_offset.unsqueeze(-1)
    new_idx = idx + idx_offset
    
    val = val.reshape(val.size(0) * val.size(1), -1)
    res = tensor.index_add(0, new_idx.view(-1), val).reshape([dim0_size, dim1_size, -1])

    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        res = res.squeeze(-1)

    return res

def batch_copy(tensor, idx, val, device=None):
    """ Do addition per row (first axis). """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]

    if device is None:
        device = tensor.device

    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = (torch.arange(0, dim0_size, device=device) * dim1_size)
    for _ in range(len(idx.size()) - 1):
        idx_offset = idx_offset.unsqueeze(-1)
    new_idx = idx + idx_offset
    
    val = val.reshape(val.size(0) * val.size(1), -1)
    res = tensor.index_copy(0, new_idx.view(-1), val).reshape([dim0_size, dim1_size, -1])

    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        res = res.squeeze(-1)

    return res

def logsumexp(
    tensor: torch.Tensor,
    dim: Union[int,Iterable] = -1, 
    keepdim: bool = False
) -> torch.Tensor:
    if type(dim) == int:
        if tensor.size(dim) == 0:
            return tensor.sum(dim=dim, keepdim=keepdim).log() # neginf
    else:
        for d in dim:
            if tensor.size(d) == 0:
                return tensor.sum(dim=dim, keepdim=keepdim).log() # neginf

    max_score = tensor.amax(dim, keepdim=True)
    stable_vec = tensor - max_score

    return max_score.sum(dim=dim, keepdim=keepdim) +\
           stable_vec.logsumexp(dim=dim, keepdim=keepdim)


def dummy_padding(vec:torch.Tensor, dummy_value: float = 0.):
    # padding one column at the beginning of the last dimension of the vector
    if vec.size(-1) != 0:
        return torch.cat([torch.full_like(vec[...,:1], dummy_value), vec], dim=-1)
    else:
        return vec.new_full(vec.size()[:-1]+(1, ), dummy_value)
    
    
def prepare_pair_embeddings(col_vecs: torch.FloatTensor, row_vecs: torch.FloatTensor):
    # Params: col_vecs: (num_col_vecs, dim_col_vec)
    #         row_vecs:    (num_row_vecs, dim_row_vec)
    # Returns : 
    # [[col_vec0:row_vec0, col_vec1:row_vec0, col_vec2:row_vec0, ...], 
    #  [col_vec0:row_vec1, col_vec1:row_vec1, col_vec2:row_vec1, ...], ...]
    # (num_row_vecs, num_col_vecs, dim_col_vec+dim_row_vec)
    if len(row_vecs.size()) == 2: # no beam size
        return torch.cat(
            [col_vecs.unsqueeze(0).expand(row_vecs.size(0), -1, -1), 
             row_vecs.unsqueeze(1).expand(-1, col_vecs.size(0), -1)], dim=-1
        )
    else:
        return torch.cat(
            [col_vecs.unsqueeze(1).expand(-1, row_vecs.size(1), -1, -1), 
             row_vecs.unsqueeze(2).expand(-1, -1, col_vecs.size(1), -1)], dim=-1
        )
    
def batched_masked_select(tensor: torch.FloatTensor, mask: torch.Tensor):
    max_len = mask.sum(dim=-1).max() # maximum number of selected elements
    mask_sorted, indices = torch.sort(
        mask.long(), descending=True, stable=True, dim=-1
    )
    mask_sorted = mask_sorted.bool()
    
    return _batched_index_select(tensor, indices)[:,:max_len], mask_sorted[:,:max_len]

def _batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    dim: int = 1,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # Input:
    #   target: (batch_size, seq_len, dim)
    #   indices: (batch_size, num_indices)
    # Returns:
    #   (batch_size, num_indices, dim)
    unidim = False
    if len(target.size()) == len(indices.size()):
        # (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
        unidim = True
        target = target.unsqueeze(-1)
    
    target_size = target.size()
    indices_size = indices.size()

    target = target.reshape(math.prod([*target_size[:dim]]), *target_size[dim:])
    indices = indices.view(math.prod([*indices_size[:dim]]), *indices_size[dim:])

    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(
            indices, target.size(1)
        )

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.reshape(-1, *target_size[dim+1:])
    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices_size) if unidim else (
        list(indices_size) + list(target_size[dim+1:]))
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.reshape(*selected_shape)
    return selected_targets

def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # Input:
    #   target: (batch_size, seq_len, dim)
    #   indices: (batch_size, num_indices)
    # Returns:
    #   (batch_size, num_indices, dim)
    unidim = False
    if len(target.size()) == len(indices.size()):
        # (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
        unidim = True
        target = target.unsqueeze(-1)
    
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(
            indices, target.size(1)
        )

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.reshape(-1, target.size(-1))
    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    if unidim:
        selected_targets = selected_targets.squeeze(-1)
    return selected_targets



def batched_index_copy(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # Input:
    #   target: (batch_size, seq_len, dim)
    #   indices: (batch_size, num_indices)
    # Returns:
    #   (batch_size, num_indices, dim)
    unidim = False
    if len(target.size()) == len(indices.size()):
        # (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
        unidim = True
        target = target.unsqueeze(-1)
    
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(
            indices, target.size(1)
        )

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))
    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    if unidim:
        selected_targets = selected_targets.squeeze(-1)
    return selected_targets

def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    # Shape: (batch_size)
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def one_hot_ignore_negative(labels, num_classes):
    return F.one_hot(
        torch.where((labels>=0), labels, num_classes), 
        num_classes=num_classes+1
    )[...,:-1].bool()


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)
    
    
def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def make_embedding(dict_size, feature_emb_size=20, std=0.02):
    # putting Embedding on the last device so it's the same with t5 last_hidden_state
    emb = nn.Embedding(
        dict_size, feature_emb_size,
        device=torch.cuda.device_count() - 1
    )
    emb.weight.data.normal_(mean=0.0, std=std)
    return emb

def make_linear(in_features, out_features, bias=True, std=0.02):
    # putting Linear on the last device so it's the same with t5 last_hidden_state
    linear = nn.Linear(
        in_features, out_features, bias,
        device=torch.cuda.device_count() - 1
    )
    linear.weight.data.normal_(mean=0.0, std=std)
    if bias:
        linear.bias.data.zero_()
    return linear

def make_ffnn(
    feat_size, hidden_size, output_size, dropout, std=0.02, activation='relu'
):
    if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
        return make_linear(feat_size, output_size, std=std)
    if not isinstance(hidden_size, Iterable):
        hidden_size = [hidden_size]

    ffnn = [make_linear(feat_size, hidden_size[0], std=std), ACT2FN[activation], dropout]
    for i in range(1, len(hidden_size)):
        ffnn += [make_linear(hidden_size[i-1], hidden_size[i], std=std), ACT2FN[activation], dropout]

    ffnn += [make_linear(hidden_size[-1], output_size, std=std)]
    return nn.Sequential(*ffnn)


def get_scheduler_lambda(scheduler_type, warmup_steps, total_steps):
    if scheduler_type == 'linear_with_warmup':
        def lambda_rule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps))
            )
        return lambda_rule
    elif scheduler_type == 'constant':
        return lambda step: 1.0
    elif scheduler_type == 'constant_with_warmup':
        return lambda step: min(1.0, float(step) / float(max(1, warmup_steps)))
    else:
        raise ValueError(f'Unknown scheduler type {scheduler_type}')
