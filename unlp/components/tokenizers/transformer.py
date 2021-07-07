# -*- coding: utf8 -*-

#
import math
from typing import Union, List

import torch.nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoTokenizer
from transformers import PreTrainedTokenizer

from unlp.common.torch_component import TorchComponent
from unlp.components.taggers.transformers.transformer_tagger import TransformerTaggingModel
from unlp.layers.transformers.encoder import TransformerEncoder
from unlp.layers.transformers.utils import build_optimizer_scheduler_with_transformer
from unlp.metrics.f1 import f1_loss
from unlp.tokenization.text import BMESTokenizationDataset
from unlp.utils.io_util import get_resource
from unlp.utils.torch_util import set_seed, clip_grad_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerComponent(TorchComponent):
    def __init__(self, **kwargs):
        super(TransformerComponent, self).__init__(**kwargs)
        self.transformer_tokenizer: Union[PreTrainedTokenizer] = None

        self.device = device

    def build_model(self, training=True, **kwargs):
        raise NotImplementedError

    def fit(self, trn_data, dev_data, save_dir, transformer, average_subwords=False, word_dropout: float = 0.2,
            hidden_dropout=None, layer_dropout=0, scalar_mix=None, grad_norm=5.0,
            transformer_grad_norm=None, lr=5e-5,
            transformer_lr=None, transformer_layers=None, gradient_accumulation=1,
            adam_epsilon=1e-8, weight_decay=0, warmup_steps=0.1, crf=False, reduction='sum',
            batch_size=32, epochs=30, patience=5, token_key=None,
            tagging_scheme='BMES', delimiter=None,
            max_seq_len=None, sent_delimiter=None, char_level=False, hard_constraint=False, transform=None, logger=None,
            seed=None,
            devices: Union[float, int, List[int]] = None, finetune=False, **kwargs):
        self._capture_config(locals())
        if not seed:
            self.config.seed = 2333
        else:
            self.config.seed = seed
        set_seed(self.config.seed)
        self.on_config_ready()

        trn_dataloader = self.build_dataloader(
            data=trn_data, batch_size=32,
            shuffle=True, tokenizer=self.transformer_tokenizer
        )
        dev_dataloader = self.build_dataloader(
            data=dev_data, batch_size=32,
            shuffle=False, tokenizer=self.transformer_tokenizer
        )
        if not finetune:
            self.model = self.build_model()
        self.model.to(device)
        criterion = self.build_criterion()
        optimizer = self.build_optimizer(
            trn=trn_dataloader,
            epochs=epochs,
            lr=lr,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            transformer_lr=transformer_lr,
        )
        self.execute_training_loop(
            trn=trn_dataloader, dev=dev_dataloader,
            epochs=epochs, criterion=criterion,
            optimizer=optimizer,
            metric=None,
            save_dir=save_dir,
            grad_norm=grad_norm,
            transformer_grad_norm=transformer_grad_norm
        )

    def build_optimizer(
            self,
            trn,
            epochs,
            lr,
            adam_epsilon,
            weight_decay,
            warmup_steps,
            transformer_lr=None,
            teacher=None,
            **kwargs
    ):

        num_training_steps = len(trn) * epochs
        if transformer_lr is None:
            transformer_lr = lr
        transformer = self.model.encoder.transformer
        optimizer, scheduler = build_optimizer_scheduler_with_transformer(
            model=self.model,
            transformer=transformer,
            lr=lr,
            transformer_lr=transformer_lr,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            adam_epsilon=adam_epsilon,
        )
        return optimizer, scheduler

    def fit_dataloader(
            self,
            trn: DataLoader,
            criterion,
            optimizer,
            metric,
            grad_norm=None,
            transformer_grad_norm=None,

    ):
        self.model.train()
        optimizer, scheduler = optimizer

        for idx, (x, y_true) in enumerate(trn):
            y_pred = self.model(*x)
            loss = self.compute_loss(criterion=criterion, y_predict=y_pred, y_true=y_true)
            loss.backward()
            self.step(optimizer=optimizer, scheduler=scheduler, grad_norm=grad_norm,
                      transformer_grad_norm=transformer_grad_norm)

    def step(self, optimizer, scheduler, grad_norm, transformer_grad_norm):
        clip_grad_norm(model=self.model, grad_norm=grad_norm,
                       transformer=self.model.encoder.transformer,
                       transformer_grad_norm=transformer_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    @torch.no_grad()
    def evaluate_dataloader(self, dev, criterion):
        self.model.eval()
        loss = 0
        for x, y_true in dev:
            y_pred = self.model(*x)
            loss += self.compute_loss(criterion=criterion, y_predict=y_pred, y_true=y_true).item()
        loss /= len(dev)
        return loss

    def compute_loss(self, criterion, y_predict, y_true):
        feature = y_predict.size()[-1]
        y_pred = y_predict.view(-1, feature)
        y_t = y_true.view(-1)
        loss = criterion(y_pred, y_t)
        f1_score = f1_loss(y_true=y_t, y_pred=y_pred, is_training=True)
        print(f'f1_score : {f1_score:.4f}, loss: {loss.item():.4f}')
        return loss

    def predict(self, save_dir):

        self.config.transformer = 'bert-base-chinese'
        self.config.word_dropout = 0.1
        self.on_config_ready()
        self.load(save_dir)
        from unlp.datasets.cws.sighan2005.pku import SIGHAN2005_PKU_DEV

        file = get_resource(SIGHAN2005_PKU_DEV)
        dataset = BMESTokenizationDataset(text_file=file, tokenizer=self.transformer_tokenizer)
        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

        for x, y in dataloader:
            y_predict = self.model(*x).argmax(axis=-1)
            print(y_predict)

    def on_config_ready(self, **kwargs):
        super(TransformerComponent, self).on_config_ready(**kwargs)
        if 'albert_chinese' in self.config.transformer:
            self.transformer_tokenizer = BertTokenizer.from_pretrained(
                self.config.transformer, use_fast=True
            )
        else:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.config.transformer, use_fast=True
            )

    def execute_training_loop(
            self, trn: DataLoader,
            dev: DataLoader,
            epochs,
            criterion,
            optimizer,
            metric,
            save_dir,
            grad_norm=0.05,
            transformer_grad_norm=None,

    ):
        min_loss = math.inf

        for epoch in range(1, epochs + 1):
            self.fit_dataloader(
                trn=trn,
                criterion=criterion,
                optimizer=optimizer,
                metric=metric,
                grad_norm=grad_norm,
                transformer_grad_norm=transformer_grad_norm
            )
            loss = self.evaluate_dataloader(
                dev=dev,
                criterion=criterion
            )
            print(f'Epoch [{epoch}], validation data loss {loss:.4f}')
            if loss < min_loss:
                print(f'current loss[{loss}] little than min_loss, save_weight to {save_dir}')
                self.save_weights(save_dir=save_dir)
                min_loss = loss


class TransformerTaggingTokenizer(TransformerComponent):
    def build_model(self, training=True, **kwargs):
        # encoder = AutoModel.from_pretrained(self.config.transformer)
        encoder = TransformerEncoder(
            transformer=self.config.transformer,
            transformer_tokenizer=self.transformer_tokenizer,
            word_dropout=self.config.word_dropout
        )
        model = TransformerTaggingModel(
            encoder=encoder,
            num_labels=5,  # 4 + pad

        )
        self.model = model
        return model

    def build_dataloader(self, data, batch_size, shuffle, **kwargs):
        file = get_resource(data)
        dataset = BMESTokenizationDataset(text_file=file, tokenizer=kwargs['tokenizer'])
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
