import torch
import numpy as np
from abc import ABC, abstractmethod, ABCMeta


class Metric(metaclass=ABCMeta):
    """
    Abstract Base class (ABC) for all Metrics.
    Taken from https://github.com/pytorch/ignite/metrics/metric.py
        and modify a bit.
    Often, data is truncated into batches. In such scenario, we call
    -   reset() in the begining of every epoch.
    -   update() after every batch
    -   compute() whenever you want to log the training/testing performance.
    """

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
    def update(self, output):
        """
        Updates the metric's state using the passed batch output.
        This is called once for each batch.
        Args:
            output: the is the output from the engine's process function
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.
        This is called at the end of each epoch.
        Returns:
            Any: the actual quantity of interest
        Raises:
            NotComputableError: raised when the metric cannot be computed
        """
        pass


class CategoricalAccuracy(Metric):
    """
    Calculates the categorical accuracy.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    def __init__(self):
        super().__init__()
        self._num_examples = 0
        self._num_correct = 0

    def reset(self):
        self._num_examples = 0
        self._num_correct = 0

    def update(self, output):
        y_pred, y = output
        _, indices = torch.max(y_pred, 1)
        correct = torch.eq(indices, y).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise ZeroDivisionError('CategoricalAccuracy must have at least'
                                    ' one example before it can be computed')
        return self._num_correct / self._num_examples


class PRMetric(Metric):
    """
    Calculates the precision and recall.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    def __init__(self, num_class=2):
        """
        precision = tp / tp + fp
        recall = tp / tp + fn
        """
        super().__init__()
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class),
                                         dtype=np.float32)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class),
                                         dtype=np.float32)

    def update(self, output):
        y_pred, y = output
        _, indices = torch.max(y_pred, 1)
        self.confusion_matrix[indices.cpu().numpy(), y.cpu().numpy()] += 1

    def compute(self):
        tp = np.diag(self.confusion_matrix)
        total_pred = np.sum(self.confusion_matrix, axis=1)  # (-1, 1)
        total_gold = np.sum(self.confusion_matrix, axis=0)  # (1, -1)
        # tn don't care
        p = np.zeros_like(total_pred)
        r = np.zeros_like(total_gold)
        for i, j in enumerate(total_pred):
            if j != 0:
                p[i] = tp[i] / j
        for i, j in enumerate(total_gold):
            if j != 0:
                r[i] = tp[i] / j
        return p, r


if __name__ == '__main__':
    # unit test
    pr = PRMetric(3)
    pr.confusion_matrix = np.array([[2, 0, 2], [0, 1, 0], [1, 0, 0]])
    print(pr.compute())
