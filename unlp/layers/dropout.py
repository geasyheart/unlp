# -*- coding: utf8 -*-

#
from typing import Sequence

import torch
from torch import nn


class WordDropout(nn.Module):
    def __init__(self, p: float, oov: int, exclude: Sequence[int], training: bool = True):
        """

        :param p: dropout 概率
        :param oov: oov id
        :param exclude: exclude ids.
        :param training:
        """
        super(WordDropout, self).__init__()
        self.p = p
        self.oov = oov
        self.exclude = exclude
        self.training = training

    @staticmethod
    def token_dropout(tokens: torch.LongTensor, oov, exclude, p, training) -> torch.LongTensor:
        """
        https://github.com/Hyperparticle/udify/blob/master/udify/models/udify_model.py

        :param tokens:
        :param oov:
        :param exclude:
        :param p:
        :param training:
        :return:
        """
        if training and p > 0:
            padding_mask = tokens.new_ones(tokens.size(), dtype=torch.bool)
            for pad in exclude:
                padding_mask &= (tokens != pad)

            dropout_mask = (tokens.new_empty(tokens.size(), dtype=torch.float).uniform_() < p)
            oov_mask = dropout_mask & padding_mask

            oov_fill = tokens.new_empty(tokens.size(), dtype=torch.long).fill_(oov)

            result = torch.where(oov_mask, oov_fill, tokens)
            return result
        else:
            return tokens

    def forward(self, tokens: torch.LongTensor):
        return self.token_dropout(
            tokens,
            self.oov,
            self.exclude,
            self.p,
            self.training
        )


class SharedDropout(nn.Module):
    """
        SharedDropout differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.
        Args:
            p (float):
                The probability of an element to be zeroed. Default: 0.5.
            batch_first (bool):
                If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
                Default: ``True``.
        Examples:
            >>> x = torch.ones(1, 3, 5)
            >>> nn.Dropout()(x)
            tensor([[[0., 2., 2., 0., 0.],
                     [2., 2., 0., 2., 2.],
                     [2., 2., 2., 2., 0.]]])
            >>> SharedDropout()(x)
            tensor([[[2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.]]])

        意思就是说：
        在y和z轴乘以相同的数值

        假设 mask 为:
        [0, 2, 2, 0, 0]

        y/z:
        [
            [1, 1,1,1,1],
            [1, 1,1,1,1],
        ]
        就变成了:

        [
            [0, 2,2,0,0],
            [0, 2,2,0, 0],
        ]

        """

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        """

        :param x:
        :return:
        """

        if not self.training:
            return x
        if self.batch_first:
            mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
        else:
            mask = self.get_mask(x[0], self.p)
        x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)
