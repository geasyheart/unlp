# -*- coding: utf8 -*-

#
import numpy as np
import pandas as pd
import tensorflow as tf

from unlp.common.keras_component import KerasComponent
from unlp.common.pd_sequence import DataFrameSequence
from unlp.datasets.ner.msra import MSRA_NER_CHAR_LEVEL_TRAIN, MSRA_NER_CHAR_LEVEL_DEV, MSRA_NER_CHAR_LEVEL_TEST
from unlp.layers.transformers.bert_tokenizer import get_tokenizer
from unlp.layers.transformers.loader_tf import build_transformer
from unlp.utils.io_util import get_resource, read_tsv_as_sentence

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# 1. F1 keras
# 2. crf layer
# 3. custom callback.

label_maps = {'O': 0, 'B-NS': 1, 'E-NS': 2, 'S-NS': 3, 'M-NS': 4, 'E-NT': 5, 'M-NT': 6, 'B-NT': 7, 'M-NR': 8, 'E-NR': 9,
              'B-NR': 10, 'S-NR': 11, 'S-NT': 12}
id_label_maps = {v: k for k, v in label_maps.items()}
tokenizer = get_tokenizer(transformer='albert_base_zh')


def init_df(file, max_len=128):
    unk_index = tokenizer.vocab['[UNK]']
    cls_index = tokenizer.vocab['[CLS]']
    sep_index = tokenizer.vocab['[SEP]']
    pad_index = tokenizer.vocab['[PAD]']
    token_ids, label_ids = [], []
    for line in read_tsv_as_sentence(file_path=file):
        tokens, labels = [], []
        for ele in line:
            token, label = ele[0].split(" ")
            tokens.append(tokenizer.vocab.get(token, unk_index))
            labels.append(label_maps[label])
        tokens = [cls_index] + tokens[:max_len - 2] + [sep_index]
        labels = [pad_index] + labels[:max_len - 2] + [pad_index]
        if len(tokens) < max_len:
            tokens = tokens + [pad_index] * (max_len - len(tokens))
            labels = labels + [pad_index] * (max_len - len(labels))
        token_ids.append(tokens)
        label_ids.append(labels)
    return pd.DataFrame({
        "tokens": token_ids,
        "labels": label_ids
    })


class CustomTransformTF(DataFrameSequence):
    def __getitem__(self, index):
        batch_df = super(CustomTransformTF, self).__getitem__(index=index)
        print(f'got {index}, cur shape is {batch_df.shape}')
        input_ids = np.array(batch_df['tokens'].to_list(), dtype=np.int32)
        labels = np.array(batch_df['labels'].to_list(), dtype=np.int32)
        token_type_ids = np.zeros(shape=input_ids.shape, dtype=np.int32)
        return [input_ids, token_type_ids], labels


class AlbertNerTF(KerasComponent):
    def __init__(self, num_labels):
        super(AlbertNerTF, self).__init__()
        self.num_labels = num_labels

    def build_model(self, *args, **kwargs) -> tf.keras.Model:
        model, tokenizer = build_transformer(
            num_labels=self.num_labels,
            transformer='albert_base_zh'
        )
        self.model = model
        return model


#
def train():
    train_data_gen = CustomTransformTF(df=init_df(file=get_resource(MSRA_NER_CHAR_LEVEL_TRAIN)))
    valid_data_gen = CustomTransformTF(df=init_df(file=get_resource(MSRA_NER_CHAR_LEVEL_DEV)))
    albert_ner = AlbertNerTF(num_labels=len(label_maps))

    albert_ner.fit(
        train_data=train_data_gen,
        dev_data=valid_data_gen,
        save_dir='/tmp/msra_albert',
        transformer='albert_base_zh',
        epochs=100,
    )


def evaluate():
    albert_ner = AlbertNerTF(num_labels=len(label_maps))
    albert_ner.build(transformer='albert_base_zh')
    albert_ner.load_weights(save_dir='/tmp/msra_albert')

    data_gen = CustomTransformTF(df=init_df(file=get_resource(MSRA_NER_CHAR_LEVEL_TEST)))
    albert_ner.model.evaluate(data_gen)


def predict():
    albert_ner = AlbertNerTF(num_labels=len(label_maps))
    albert_ner.build(transformer='albert_base_zh')
    albert_ner.load_weights(save_dir='/tmp/msra_albert')

    ys = []
    for x, y in CustomTransformTF(df=init_df(file=get_resource(MSRA_NER_CHAR_LEVEL_TEST))):
        y_predict = albert_ner.model.predict_on_batch(x).argmax(axis=-1)
        for _predict in y_predict:
            sample_predict = [id_label_maps[_] for _ in _predict]
            ys.append(sample_predict)
    return ys


if __name__ == '__main__':
    # train()
    evaluate()
