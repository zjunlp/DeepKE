import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothSoftmaxCEV1(nn.Module):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


def taylor_softmax_v1(x, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log: out = out.log()
    return out

class LogTaylorSoftmaxV1(nn.Module):
    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV1, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n
    def forward(self, x):
        return taylor_softmax_v1(x, self.dim, self.n, use_log=True)


class TaylorCrossEntropyLossV1(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV1, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV1(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
                ignore_index=self.ignore_index)
        return loss
