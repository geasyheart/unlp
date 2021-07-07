# -*- coding: utf8 -*-

#
import torch
from torch import nn
from transformers import PreTrainedTokenizer, AutoModel

from unlp.layers.dropout import WordDropout


class TransformerEncoder(nn.Module):
    """"""

    def __init__(self, transformer, transformer_tokenizer, word_dropout=None):
        super(TransformerEncoder, self).__init__()

        self.transformer: str = transformer
        self.transformer_tokenizer: PreTrainedTokenizer = transformer_tokenizer
        self.word_dropout = word_dropout

        if word_dropout:
            oov = self.transformer_tokenizer.mask_token_id
            exclude = (
                self.transformer_tokenizer.cls_token_id,
                self.transformer_tokenizer.sep_token_id,
                self.transformer_tokenizer.pad_token_id
            )
            self.word_dropout = WordDropout(p=word_dropout, oov=oov, exclude=exclude)
        else:
            self.word_dropout = None
        transformer = AutoModel.from_pretrained(transformer)

        if hasattr(transformer, 'encoder') and hasattr(transformer, 'decoder'):
            transformer = transformer.encoder
        self.transformer = transformer

    def forward(self, input_ids: torch.LongTensor, token_type_ids: torch.LongTensor):
        if self.word_dropout:
            input_ids = self.word_dropout(input_ids)
        attention_mask = input_ids.ne(self.transformer_tokenizer.pad_token_id)
        if self.transformer.config.output_hidden_states:
            outputs = self.transformer(input_ids, attention_mask, token_type_ids)[-1]
        else:
            outputs = self.transformer(input_ids, attention_mask, token_type_ids)[0]
        return outputs
