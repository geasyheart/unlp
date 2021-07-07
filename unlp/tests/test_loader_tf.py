# -*- coding: utf8 -*-

#


from unlp.layers.transformers.loader_tf import build_transformer

# albert
# model, tokenizer = build_transformer(transformer='albert_base_zh', max_seq_len=128, num_labels=4, )
# tokens = tokenizer.tokenize('我是中国人')
# print(tokens, tokenizer.convert_tokens_to_ids(tokens), tokenizer.convert_tokens_to_ids(['[CLS]']))
# model.summary()

model, tokenizer = build_transformer(transformer='chinese_L-12_H-768_A-12', max_seq_len=128, num_labels=4, )
tokens = tokenizer.tokenize('我是中国人')
print(tokens, tokenizer.convert_tokens_to_ids(tokens), tokenizer.convert_tokens_to_ids(['[CLS]']))
model.summary()
