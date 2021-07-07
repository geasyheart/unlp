# -*- coding: utf8 -*-

#
from abc import ABC
from typing import Set

import torch
from torch.nn import functional as F

from .metric import Metric


class F1(Metric, ABC):
    """"""

    def __init__(self, nb_pred=0, nb_true=0, nb_correct=0):
        super(F1, self).__init__()
        self.nb_pred = nb_pred
        self.nb_true = nb_true
        self.nb_correct = nb_correct

    def __repr__(self):
        p, r, f = self.prf

        return f"P: {p:.2%} R: {r:.2%} F1: {f:.2%}"

    @property
    def prf(self):
        nb_correct = self.nb_correct
        nb_pred = self.nb_pred
        nb_true = self.nb_true

        p = nb_correct / nb_pred if nb_pred > 0 else 0.
        r = nb_correct / nb_true if nb_true > 0 else 0.
        f = 2 * p * r / (p + r) if p + r > 0 else 0.
        return p, r, f

    @property
    def score(self):
        return self.prf[-1]

    def __call__(self, pred: Set, gold: Set, mask=None):
        super(F1, self).__call__(pred=pred, gold=gold, mask=mask)

        self.nb_correct += len(pred & gold)
        self.nb_pred += len(pred)
        self.nb_true += len(gold)


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_pred.ndim == 2
    assert y_true.ndim == 1
    y_true = F.one_hot(y_true, y_pred.size()[-1]).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)

    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)
    return 1 - f1.mean()
