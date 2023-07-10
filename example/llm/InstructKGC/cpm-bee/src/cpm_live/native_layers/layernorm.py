import torch


@torch.jit.script  # type: ignore
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class LayerNorm(torch.nn.Module):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        dtype: torch.dtype = torch.half,
        eps: float = 1e-6,
        init_var: float = 1.0,
    ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = torch.nn.parameter.Parameter(torch.full((dim_norm,), init_var, dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.size(-1) == self.dim_norm
        return rms_layernorm(x, self.weight, self.eps)
