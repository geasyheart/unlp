# -*- coding: utf8 -*-

#
import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict

import torch.nn

from unlp.metrics.f1 import F1
from unlp.utils import isdebugging
from unlp.utils.io_util import get_resource, merge_dict
from unlp.utils.log_util import logger
from unlp.utils.structure import SerializableDict
from unlp.utils.torch_util import cuda_devices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TorchComponent(ABC):
    def __init__(self, **kwargs) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.config = SerializableDict(**kwargs)

    def _capture_config(self, locals_: Dict,
                        exclude=(
                                'trn_data', 'dev_data', 'save_dir', 'kwargs', 'self', 'logger', 'verbose',
                                'dev_batch_size', '__class__', 'devices', 'eval_trn')):
        if 'kwargs' in locals_:
            locals_.update(locals_['kwargs'])
        locals_ = dict((k, v) for k, v in locals_.items() if k not in exclude and not k.startswith('_'))
        self.config.update(locals_)
        return self.config

    @property
    def model_(self) -> torch.nn.Module:
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def save_weights(self, save_dir, filename='model.pt', trainable_only=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model = self.model_
        state_dict = model.state_dict()
        if trainable_only:
            trainable_names = set(n for n, p in model.named_parameters() if p.requires_grad)
            state_dict = dict((n, p) for n, p in state_dict.items() if n in trainable_names)
        torch.save(state_dict, os.path.join(save_dir, filename))

    def load_weights(self, save_dir, filename='model.pt'):
        save_dir = get_resource(save_dir)
        filename = os.path.join(save_dir, filename)
        self.model_.load_state_dict(
            torch.load(filename, map_location='cpu'),
            strict=False
        )

    @property
    def devices(self):
        if self.model is None: return None
        if hasattr(self.model, 'device_ids'):
            return self.model.device_ids
        device: torch.device = next(self.model.parameters()).device
        return [device]

    def load(self, save_dir: str, devices=None, **kwargs):
        save_dir = get_resource(save_dir)

        self.model = self.build_model(
            **merge_dict(self.config, training=False, **kwargs, overwrite=True,
                         inplace=True))

        self.load_weights(save_dir)

        if devices is None and self.model:
            devices = self.devices

        # self.to(devices)
        self.model.to(device)
        self.model.eval()

    def to(
            self,
            devices=Union[int, float, List[int], Dict[str, Union[int, torch.device]]],
    ):

        if devices == -1 or devices == [-1]:
            devices = []
        elif isinstance(devices, (int, float)) or devices is None:
            devices = cuda_devices(devices)
        if devices:
            logger.info(f'Using GPUs: [on_blue][cyan][bold]{devices}[/bold][/cyan][/on_blue]')
            if isinstance(devices, list):
                logger.info(f'Moving model to GPUs {devices} [blink][yellow]...[/yellow][/blink]')
                self.model = self.model.to(devices[0])
                if len(devices) > 1 and not isdebugging() and not isinstance(self.model, torch.nn.DataParallel):
                    self.model = self.parallelize(devices)
            elif isinstance(devices, dict):
                for name, module in self.model.named_modules():
                    for regex, device in devices.items():
                        try:
                            on_device: torch.device = next(module.parameters()).device
                        except StopIteration:
                            continue
                        if on_device == device:
                            continue
                        if isinstance(device, int):
                            if on_device.index == device:
                                continue
                        if re.match(regex, name):
                            if not name:
                                name = '*'
                            logger.info(f'Moving module [yellow]{name}[/yellow] to [on_yellow][magenta][bold]{device}'
                                        f'[/bold][/magenta][/on_yellow]: [red]{regex}[/red]\n')
                            module.to(device)
            else:
                raise ValueError(f'Unrecognized devices {devices}')
        else:
            if logger:
                logger.info('Using [red]CPU[/red]')

    @abstractmethod
    def build_model(self, training=True, **kwargs):
        raise NotImplementedError

    def parallelize(self, devices: List[Union[int, torch.device]]):
        return torch.nn.DataParallel(self.model, device_ids=devices)

    def on_config_ready(self, **kwargs):
        pass

    def fit(
            self,
            trn_data,
            dev_data,
            save_dir,
            batch_size,
            epochs,
            devices=None,
            seed=None,
            finetune: bool = False,
            eval_trn=True,
            **kwargs
    ):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def build_dataloader(
            self, data, batch_size,
            shuffle,
            **kwargs):
        pass

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def build_metric(self, **kwargs):
        return F1()

    def build_optimizer(self, **kwargs):
        pass
