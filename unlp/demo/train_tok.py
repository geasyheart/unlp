# -*- coding: utf8 -*-

#
from unlp.components.tokenizers.transformer import TransformerTaggingTokenizer

tokenizer = TransformerTaggingTokenizer()

save_dir = '/tmp/tok/'
# tokenizer.fit(
#     trn_data=SIGHAN2005_PKU_TRAIN_ALL,
#     dev_data=SIGHAN2005_PKU_TEST,
#     save_dir=save_dir,
#     transformer='bert-base-chinese',
#     max_seq_len=128,
#     char_level=True,
#     hard_constraint=True,
#     epochs=3,
#     adam_epsilon=1e-6,
#     warmup_steps=0.1,
#     weight_decay=0.01,
#     word_dropout=0.1,
#     seed=12332123,
#
# )

tokenizer.predict(save_dir=save_dir)
print(f'Model saved in {save_dir}')
