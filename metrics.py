import torch
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.metrics import precision_recall_fscore_support


class Metric(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the metric to to it's initial state.
        This is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, *args):
        """
        Updates the metric's state using the passed batch output.
        This is called once for each batch.
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.
        This is called at the end of each epoch.
        :return: the actual quantity of interest
        """
        pass


class PRMetric():
    def __init__(self):
        """
        暂时调用 sklearn 的方法
        """
        self.y_true = np.empty(0)
        self.y_pred = np.empty(0)

    def reset(self):
        """
        重置为0
        """
        self.y_true = np.empty(0)
        self.y_pred = np.empty(0)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        更新tensor，保留值，取消原有梯度
        """
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.argmax(y_pred, axis=-1)

        self.y_true = np.append(self.y_true, y_true)
        self.y_pred = np.append(self.y_pred, y_pred)

    def compute(self):
        """
        计算acc,p,r,f1并返回
        """
        p, r, f1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='macro', warn_for=tuple())
        _, _, acc, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='micro', warn_for=tuple())

        return acc, p, r, f1
