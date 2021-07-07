# -*- coding: utf8 -*-

#
from typing import Any, Tuple

import torch
from torch import nn

from unlp.components.parsers.biaffine.biaffine import Biaffine
from unlp.components.parsers.biaffine.mlp import MLP
from unlp.layers.transformers.token_embedding import TransformerEmbedding


class BiaffineDecoder(nn.Module):
    def __init__(self, hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, n_rels, arc_dropout=None,
                 rel_dropout=None) -> None:
        super().__init__()
        # the MLP layers
        self.mlp_arc_h = MLP(hidden_size,
                             n_mlp_arc,
                             dropout=arc_dropout or mlp_dropout)
        self.mlp_arc_d = MLP(hidden_size,
                             n_mlp_arc,
                             dropout=arc_dropout or mlp_dropout)
        self.mlp_rel_h = MLP(hidden_size,
                             n_mlp_rel,
                             dropout=rel_dropout or mlp_dropout)
        self.mlp_rel_d = MLP(hidden_size,
                             n_mlp_rel,
                             dropout=rel_dropout or mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)

    def forward(self, x, mask=None, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        arc_d, arc_h, rel_d, rel_h = self.apply_mlps(x)

        s_arc, s_rel = self.decode(arc_d, arc_h, rel_d, rel_h, mask, self.arc_attn, self.rel_attn)

        return s_arc, s_rel

    @staticmethod
    def decode(arc_d, arc_h, rel_d, rel_h, mask, arc_attn, rel_attn):
        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        if mask is not None:
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        return s_arc, s_rel

    def apply_mlps(self, x):
        # apply MLPs to the hidden states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)
        return arc_d, arc_h, rel_d, rel_h


class BiaffineDependencyModel(nn.Module):
    def __init__(self, config):
        super(BiaffineDependencyModel, self).__init__()
        self.encoder = TransformerEmbedding(
            model=config.transformer,
            n_layers=config.get('n_bert_layers', 4),
            requires_grad=True,
        )
        self.biaffine_decoder = BiaffineDecoder(self.encoder.hidden_size,
                                                config.n_mlp_arc,
                                                config.n_mlp_rel,
                                                config.mlp_dropout,
                                                config.n_rels)

    def forward(self, words):
        x = self.encoder(subwords=words)

        mask = words.ne(0) if len(words.shape) < 3 else words.ne(0).any(-1)

        # mask = x.ne(self.args.pad_index) if len(x.shape) < 3 else x.ne(self.args.pad_index).any(-1)
        s_arc, s_rel = self.biaffine_decoder(x, mask=mask)
        return s_arc, s_rel
