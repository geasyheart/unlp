# -*- coding: utf8 -*-

#

from torch import nn

from unlp.layers.transformers.encoder import TransformerEncoder


class TransformerTaggingModel(nn.Module):
    def __init__(
            self,
            encoder: TransformerEncoder,
            num_labels,
            crf=False,
            secondary_encoder=None
    ) -> None:
        super(TransformerTaggingModel, self).__init__()
        self.encoder = encoder
        self.secondary_encoder = secondary_encoder
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, token_type_ids):
        x = self.encoder(input_ids, token_type_ids=token_type_ids)

        x = self.classifier(x)
        return x
