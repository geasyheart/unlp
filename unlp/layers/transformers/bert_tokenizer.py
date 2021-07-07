# -*- coding: utf8 -*-

#
from unlp.common.tokenizer import Tokenizer
from .loader_tf import build_transformer


def get_tokenizer(transformer: str, with_custom_tokenizer=False):
    _, tokenizer = build_transformer(num_labels=0, transformer=transformer, tokenizer_only=True)
    if with_custom_tokenizer:
        return Tokenizer(token_dict=tokenizer.vocab)
    return tokenizer
