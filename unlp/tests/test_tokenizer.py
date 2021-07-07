# -*- coding: utf8 -*-

#
from unlp.common.tokenizer import Tokenizer
from unlp.layers.transformers.loader_tf import build_transformer

_, tokenizer = build_transformer(num_labels=0, transformer='albert_base_zh', tokenizer_only=True)

t = Tokenizer(tokenizer.vocab)

for word in ('你好', "hello world"):
    res = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
    res2 = t.encode(first=word)
    print(res, res2)
