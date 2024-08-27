import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CNN(nn.Module):
    """
    nlp 里为了保证输出的句长 = 输入的句长，一般使用奇数 kernel_size，如 [3, 5, 7, 9]
    当然也可以不等长输出，keep_length 设为 False
    此时，padding = k // 2
    stride 一般为 1
    """
    def __init__(self, config):
        """
        in_channels      : 一般就是 word embedding 的维度，或者 hidden size 的维度
        out_channels     : int
        kernel_sizes     : list 为了保证输出长度=输入长度，必须为奇数: 3, 5, 7...
        activation       : [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]
        pooling_strategy : [max, avg, cls]
        dropout:         : float
        """
        super(CNN, self).__init__()

        # self.xxx = config.xxx
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_sizes = config.kernel_sizes
        self.activation = config.activation
        self.pooling_strategy = config.pooling_strategy
        self.dropout = config.dropout
        self.keep_length = config.keep_length
        for kernel_size in self.kernel_sizes:
            assert kernel_size % 2 == 1, "kernel size has to be odd numbers."

        # convolution
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      stride=1,
                      padding=k // 2 if self.keep_length else 0,
                      dilation=1,
                      groups=1,
                      bias=False) for k in self.kernel_sizes
        ])

        # activation function
        assert self.activation in ['relu', 'lrelu', 'prelu', 'selu', 'celu', 'gelu', 'sigmoid', 'tanh'], \
            'activation function must choose from [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]'
        self.activations = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['selu', nn.SELU()],
            ['celu', nn.CELU()],
            ['gelu', GELU()],
            ['sigmoid', nn.Sigmoid()],
            ['tanh', nn.Tanh()],
        ])

        # pooling
        assert self.pooling_strategy in ['max', 'avg', 'cls'], 'pooling strategy must choose from [max, avg, cls]'

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        """
            :param x: torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H] 一般是经过embedding后的值
            :param mask: [batch_size, max_len], 句长部分为0，padding部分为1。不影响卷积运算，max-pool一定不会pool到pad为0的位置
            :return:
            """
        # [B, L, H] -> [B, H, L] （注释：将 H 维度当作输入 channel 维度)
        x = torch.transpose(x, 1, 2)

        # convolution + activation  [[B, H, L], ... ]
        act_fn = self.activations[self.activation]

        x = [act_fn(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)

        # mask
        if mask is not None:
            # [B, L] -> [B, 1, L]
            mask = mask.unsqueeze(1)
            x = x.masked_fill_(mask, 1e-12)

        # pooling
        # [[B, H, L], ... ] -> [[B, H], ... ]
        if self.pooling_strategy == 'max':
            xp = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            # 等价于 xp = torch.max(x, dim=2)[0]

        elif self.pooling_strategy == 'avg':
            x_len = mask.squeeze().eq(0).sum(-1).unsqueeze(-1).to(torch.float).to(device=mask.device)
            xp = torch.sum(x, dim=-1) / x_len

        else:
            # self.pooling_strategy == 'cls'
            xp = x[:, :, 0]

        x = x.transpose(1, 2)
        x = self.dropout(x)
        xp = self.dropout(xp)

        return x, xp  # [B, L, Hs], [B, Hs]
