import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import MultiHeadAttention


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


class TransformerAttention(nn.Module):
    def __init__(self, config):
        super(TransformerAttention, self).__init__()

        # self.xxx = config.xxx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.output_attentions = config.output_attentions
        self.layer_norm_eps = config.layer_norm_eps

        self.multihead_attention = MultiHeadAttention(self.hidden_size, self.num_heads, self.dropout,
                                                      self.output_attentions)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.layerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(self, x, key_padding_mask=None, attention_mask=None, head_mask=None):
        """
        :param x: [B, L, Hs]
        :param attention_mask: [B, L] padding后的句子后面补0了，补0的位置为True，前面部分为False
        :param head_mask: [L] [N,L]
        :return:
        """
        attention_outputs = self.multihead_attention(x, x, x, key_padding_mask, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        attention_output = self.dense(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.layerNorm(attention_output + x)
        outputs = (attention_output, ) + attention_outputs[1:]  # 后面是 attention weight
        return outputs


class TransformerOutput(nn.Module):
    def __init__(self, config):
        super(TransformerOutput, self).__init__()

        # self.xxx = config.xxx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout = config.dropout
        self.layer_norm_eps = config.layer_norm_eps

        self.zoom_in = nn.Linear(self.hidden_size, self.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.zoom_out = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.layerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(self, input_tensor):
        hidden_states = self.zoom_in(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.zoom_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()

        self.attention = TransformerAttention(config)
        self.output = TransformerOutput(config)

    def forward(self, hidden_states, key_padding_mask=None, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, key_padding_mask, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        layer_output = self.output(attention_output)
        outputs = (layer_output, ) + attention_outputs[1:]
        return outputs


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        # self.xxx = config.xxx
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.layer = nn.ModuleList([TransformerLayer(config) for _ in range(self.num_hidden_layers)])

    def forward(self, hidden_states, key_padding_mask=None, attention_mask=None, head_mask=None):
        """
        :param hidden_states: [B, L, Hs]
        :param key_padding_mask: [B, S]                   为 1/True 的地方需要 mask
        :param attn_mask: [S] / [L, S] 指定位置 mask 掉，   为 1/True 的地方需要 mask
        :param head_mask: [N] / [L, N] 指定 head mask 掉， 为 1/True 的地方需要 mask
        """
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.expand((self.num_hidden_layers, ) + head_mask.shape)
        else:
            head_mask = [None] * self.num_hidden_layers

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_outputs = layer_module(hidden_states, key_padding_mask, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions, )
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
