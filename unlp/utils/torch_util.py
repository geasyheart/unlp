# -*- coding: utf8 -*-

#
import os
import random
from typing import Dict, List

import numpy.random
import torch
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from torch import nn


def gpus_available() -> Dict[int, float]:
    try:
        nvmlInit()
        gpus = {}
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_devices is None:
            visible_devices = list(range(nvmlDeviceGetCount()))
        else:
            visible_devices = {int(x.strip()) for x in visible_devices.split(',')}
        for i, real_id in enumerate(visible_devices):
            h = nvmlDeviceGetHandleByIndex(real_id)
            info = nvmlDeviceGetMemoryInfo(h)
            total = info.total
            free = info.free
            ratio = free / total
            gpus[i] = ratio
            # print(f'total    : {info.total}')
            # print(f'free     : {info.free}')
            # print(f'used     : {info.used}')
            # t = torch.cuda.get_device_properties(0).total_memory
            # c = torch.cuda.memory_cached(0)
            # a = torch.cuda.memory_allocated(0)
            # print(t, c, a)
        nvmlShutdown()
        return dict(sorted(gpus.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        return dict((i, 1.0) for i in range(torch.cuda.device_count()))


def cuda_devices(query=None) -> List[int]:
    """Decide which GPUs to use

    Args:
      query:  (Default value = None)

    Returns:


    """
    if isinstance(query, list):
        if len(query) == 0:
            return [-1]
        return query
    if query is None:
        query = gpus_available()
        if not query:
            return []
        size, idx = max((v, k) for k, v in query.items())
        # When multiple GPUs have the same size, randomly pick one to avoid conflicting
        gpus_with_same_size = [k for k, v in query.items() if v == size]
        query = random.choice(gpus_with_same_size)
    if isinstance(query, float):
        gpus = gpus_available()
        if not query:
            return []
        query = [k for k, v in gpus.items() if v > query]
    elif isinstance(query, int):
        query = [query]
    return query


def set_seed(seed=2333, dont_care_speed=False):
    """
    Copied from https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L76
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if dont_care_speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clip_grad_norm(model: nn.Module, grad_norm, transformer: nn.Module = None, transformer_grad_norm=None):
    if transformer_grad_norm is None:
        if grad_norm is not None:
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_norm)
    else:
        is_transformer = []
        non_transformer = []
        transformer = set(transformer.parameters())
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p in transformer:
                is_transformer.append(p)
            else:
                non_transformer.append(p)
        nn.utils.clip_grad_norm_(non_transformer, grad_norm)
        nn.utils.clip_grad_norm_(is_transformer, transformer_grad_norm)
