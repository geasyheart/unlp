# -*- coding: utf8 -*-

#
import json

import torch
from torch.utils.data import DataLoader

from unlp.metrics.metric import AttachmentMetric
from unlp.tokenization.dep import DependencyDataset
from unlp.utils.io_util import get_resource
from .biaffine_model import BiaffineDependencyModel
from ...tokenizers.transformer import TransformerComponent


class BiaffineDependency(TransformerComponent):
    def build_model(self, training=True, **kwargs):
        with open('rel_ids.json', 'r') as f:
            n_rels = len(json.loads(f.read()))
            self.config.update({'n_rels': n_rels})
        model = BiaffineDependencyModel(
            config=self.config
        )
        self.model = model
        return model

    def build_dataloader(
            self, data, batch_size,
            shuffle,
            **kwargs):
        file = get_resource(data)
        dataset = DependencyDataset(
            file=file,
            tokenizer=self.transformer_tokenizer,
            batch_size=batch_size
        )
        return dataset.dataloader()

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

        for idx, (texts, words, heads, rels) in enumerate(trn):
            subwords = words.to(self.device)
            arc_score, rel_score = self.model(words=subwords)
            word_mask = subwords.ne(0)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0

            arcs, rels = heads.to(self.device), rels.to(self.device)

            loss = self.compute_loss(arc_scores=arc_score, rel_scores=rel_score, arcs=arcs, rels=rels, mask=mask,
                                     criterion=criterion)
            loss.backward()

            self.step(optimizer=optimizer, scheduler=scheduler, grad_norm=grad_norm,
                      transformer_grad_norm=transformer_grad_norm)

    def compute_loss(self, arc_scores, rel_scores, arcs, rels, mask, criterion, batch=None):

        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        arc_loss = criterion(arc_scores, arcs)
        rel_loss = criterion(rel_scores, rels)
        loss = arc_loss + rel_loss

        return loss

    @torch.no_grad()
    def evaluate_dataloader(self, dev, criterion):
        self.model.eval()
        total_loss = 0
        metric = AttachmentMetric()
        for idx, (texts, words, heads, rels) in enumerate(dev):
            subwords = words.to(self.device)
            arc_score, rel_score = self.model(words=subwords)
            word_mask = subwords.ne(0)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0

            arcs, rels = heads.to(self.device), rels.to(self.device)

            total_loss += self.compute_loss(arc_scores=arc_score, rel_scores=rel_score, arcs=arcs, rels=rels, mask=mask,
                                            criterion=criterion)

            arc_preds = arc_score.argmax(-1)
            rel_preds = rel_score.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
            # print(len(texts[0]), heads[0], arc_pred[0], rels[0], rel_pred[0])
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(dev)
        print(f'Metric : {metric}')
        return total_loss

    @torch.no_grad()
    def predict(self, save_dir, data, **kwargs):
        """

        :param save_dir:
        :param data:
        :param kwargs:
        :return:
        """
        self._capture_config(locals())
        self.on_config_ready()
        self.load(save_dir)

        dataloader = self.build_dataloader(data=data, batch_size=2, shuffle=False)
        for idx, (texts, words, heads, rels) in enumerate(dataloader):
            subwords = words.to(self.device)
            arc_score, rel_score = self.model(words=subwords)
            word_mask = subwords.ne(0)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0
            lens = mask.sum(1)
            arcs, rels = heads.to(self.device), rels.to(self.device)

            arc_pred = arc_score.argmax(-1)
            rel_pred = rel_score.argmax(-1).gather(-1, arc_pred.unsqueeze(-1)).squeeze(-1)

            arc_pred_res = arc_pred[mask].split(lens.tolist())
            rel_pred_res = rel_pred[mask].split(lens.tolist())

            yield arc_pred_res, rel_pred_res
