# -*- coding: utf8 -*-

#
# https://github.com/kpe/bert-for-tf2
import glob
import os
from typing import Tuple, Union

import bert
from bert import bert_models_google
from bert.loader_albert import albert_models_google
from bert.tokenization import albert_tokenization
from bert.tokenization import bert_tokenization
from tensorflow import keras

from unlp.utils.io_util import get_resource


def build_transformer(
        num_labels: int,
        transformer: str = 'albert_base_zh',
        max_seq_len: int = 128,
        tokenizer_only: bool = False,
        tagging: bool = True
) -> Tuple[
    Union[None, keras.Model],
    Union['albert_tokenization.FullTokenizer', 'bert_tokenization.FullTokenizer']
]:
    if transformer in albert_models_google:
        from bert.tokenization.albert_tokenization import FullTokenizer
        model_url = albert_models_google[transformer]
        albert = True
    elif transformer in bert_models_google:
        from bert.tokenization.bert_tokenization import FullTokenizer
        model_url = bert_models_google[transformer]
        albert = False
    else:
        raise ValueError(f'Unknown model {transformer}')
    bert_dir = get_resource(model_url)
    vocab = glob.glob(os.path.join(bert_dir, '*vocab*.txt'))

    assert len(vocab) == 1, 'vocab found error.'

    vocab = vocab[0]
    lower_case = any(key in transformer for key in ['uncased', 'multilingual', 'chinese', 'albert'])

    tokenizer = FullTokenizer(vocab_file=vocab, do_lower_case=lower_case)
    if tokenizer_only:
        return None, tokenizer

    bert_params = bert.params_from_pretrained_ckpt(bert_dir)

    l_bert = bert.BertModelLayer.from_params(bert_params, name="albert" if albert else "bert")

    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='input_ids')
    l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='token_type_ids')

    output = l_bert([l_input_ids, l_token_type_ids])

    if not tagging:
        # get cls
        output = keras.layers.Lambda(lambda x: x[:, 0, :])(output)

    # dropout
    if bert_params.hidden_dropout:
        output = keras.layers.Dropout(bert_params.hidden_dropout, name='hidden_dropout')(output)
    if num_labels == 1:
        logits = keras.layers.Dense(num_labels, activation='sigmoid',
                                    kernel_initializer=keras.initializers.TruncatedNormal(
                                        bert_params.initializer_range
                                    ))(output)
    else:
        logits = keras.layers.Dense(num_labels, activation='softmax',
                                    kernel_initializer=keras.initializers.TruncatedNormal(
                                        bert_params.initializer_range
                                    ))(output)

    model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    ckpt = glob.glob(os.path.join(bert_dir, '*.index'))

    model_ckpt, _ = os.path.splitext(ckpt[0])

    if albert:
        skipped_weight_value_tuples = bert.load_stock_weights(l_bert, model_ckpt)
    else:
        skipped_weight_value_tuples = bert.load_bert_weights(l_bert, model_ckpt)

    assert len(skipped_weight_value_tuples) == 0, f'failed to load pretrained {transformer}'

    return model, tokenizer
