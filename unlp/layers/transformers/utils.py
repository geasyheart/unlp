# -*- coding: utf8 -*-

#
from collections import defaultdict
from typing import Union

import torch.nn
from transformers import optimization


def build_optimizer_scheduler_with_transformer(
        model: torch.nn.Module,
        transformer: torch.nn.Module,
        lr: float,
        transformer_lr: float,
        num_training_steps: int,
        warmup_steps: Union[float, int],

        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-6,
        no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight')

):
    """


    :param warmup_steps:
    :param model:
    :param transformer:
    :param lr:
    :param transformer_lr:
    :param num_training_steps:
    :param weight_decay:
    :param adam_epsilon:
    :param no_decay:
    :return:
    """
    optimizer = build_optimizer_for_pretrained(
        model=model,
        pretrained=transformer,
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_epsilon,
        transformer_lr=transformer_lr,
        no_decay=no_decay,

    )

    if isinstance(warmup_steps, float):
        assert 0 < warmup_steps < 1, 'warmup_steps between 0 , 1'
        warmup_steps = num_training_steps * warmup_steps
    scheduler = optimization.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def build_optimizer_for_pretrained(
        model: torch.nn.Module,
        pretrained: torch.nn.Module,
        lr: float,
        weight_decay=0.01,
        eps=1e-8,
        transformer_lr=None,
        transformer_weight_decay=None,
        no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
        **kwargs
):
    if transformer_lr is None:
        transformer_lr = lr
    if transformer_weight_decay is None:
        transformer_weight_decay = weight_decay
    params = defaultdict(lambda: defaultdict(list))
    pretrained_parameters = set(pretrained.parameters())

    if isinstance(no_decay, tuple):
        def no_decay_fn(name):
            return any(nd in name for nd in no_decay)
    else:
        assert callable(no_decay)
        no_decay_fn = no_decay

    for n, p in model.named_parameters():
        is_pretrained = 'pretrained' if p in pretrained_parameters else 'no_pretrained'
        is_no_decay = 'no_decay' if no_decay_fn(n) else 'decay'
        params[is_pretrained][is_no_decay].append(p)
    # https://zhuanlan.zhihu.com/p/56386373
    # https://github.com/lonePatient/BERT-NER-Pytorch
    grouped_parameters = [
        {'params': params['pretrained']['decay'], 'weight_decay': transformer_weight_decay, 'lr': transformer_lr},
        {'params': params['pretrained']['no_decay'], 'weight_decay': 0., 'lr': transformer_lr},
        {'params': params['no_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
        {'params': params['no_pretrained']['no_decay'], 'weight_decay': 0., 'lr': lr},
    ]
    return optimization.AdamW(
        params=grouped_parameters,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        **kwargs
    )
