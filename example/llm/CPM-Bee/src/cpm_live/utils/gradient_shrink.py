import torch


class OpGradientShrink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


def gradient_shrink(x: torch.Tensor, alpha: float = 0.1):
    return OpGradientShrink.apply(x, alpha)
