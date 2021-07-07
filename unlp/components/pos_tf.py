# -*- coding: utf8 -*-

#
import json
import os
from typing import Dict

import tensorflow as tf

from unlp.common.keras_component import KerasComponent
from unlp.datasets.pos import ctb5
from unlp.layers.embeddings.fasttext_tf import build_fasttext_embedding
from unlp.utils.io_util import get_resource, read_tsv_as_sentence


# tf.config.experimental_run_functions_eagerly(True)


# ############## 直接使用默认embedding ###############
# 1. embedding, without mask_zero   # accuracy: 0.9830
# 2. embedding, with mask_zero  # 由于<pad>我给的索引就是0,所以直接mask掉.

# 但是对于2，针对相应padding位置，得出来的预测结果是1,而不是0,所以此处的mask_zero的真实含义没有明白,
# 但是对于结果预测并不影响，因为padding 长度是知道的，只需要把padding长度去掉不就可以了么。

# mask_zero作用：
# 1. 对于后续层，比如lstm，会产生影响，因为padding value(比如0)会加入运算，如果告诉mask掉了，那么肯定不会产生影响
# 2. 训练速度（不过个人感觉这个不应该算是主要优势）

# 上述两种方法的learn_rate在2e-4即可。

# ############## 使用fasttext 作为 embedding. ###############
# 3. fasttext embedding, with mask_zero , but current mask value is <pad>      # ignore
# 4. fasttext embedding, with mask_zero and modify mask function padding
# 5. fasttext embedding, with mask_zero and padding use 0, not <pad>

# fasttext embedding,最终做法就是针对除了<pad>标识为0,然后进行训练，但是最终效果没有直接上embedding效果好.
# 最后更新， trainable=False去掉进行微调，结果还是不错的

class RNNPartOfSpeechTaggerTF(KerasComponent):
    def __init__(self, model_path: str, word_indices: Dict[str, int], units: int):
        super(RNNPartOfSpeechTaggerTF, self).__init__()
        self.fasttext_model_path = model_path
        self.word_indices = word_indices
        self.units = units

    def build_model(self, *args, **kwargs) -> tf.keras.Model:
        input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        embedding = build_fasttext_embedding(model_path=self.fasttext_model_path, words_indices=self.word_indices)
        output = embedding(input)
        output = tf.keras.layers.Dropout(0.3)(output)
        output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, activation='relu', return_sequences=True))(output)

        output = tf.keras.layers.Dropout(0.3)(output)
        output = tf.keras.layers.Dense(units=self.units, activation='softmax')(output)

        model = tf.keras.Model(input, output)

        # model = tf.keras.Sequential()
        # embedding = build_fasttext_embedding(model_path=self.fasttext_model_path, words_indices=self.word_indices)
        # # embedding = tf.keras.layers.Embedding(input_dim=len(self.word_indices), output_dim=300)
        # model.add(embedding)
        # model.add(tf.keras.layers.Dropout(0.3))
        # model.add(
        #     tf.keras.layers.Bidirectional(
        #         tf.keras.layers.LSTM(units=64, activation='relu',  return_sequences=True)
        #     )
        # )
        # model.add(tf.keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.Dense(units=self.units, activation='softmax'))
        return model

    def build(self, *args, **kwargs):
        model = self.build_model()
        optimizer = self.build_optimizer()
        loss = self.build_loss()
        metrics = self.build_metrics()
        # if not self.model.built:
        #     pass
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.summary()
        self.model = model
        return model, optimizer, loss, metrics

    def fit(self, x, y, valid_x, valid_y, save_dir, *args, **kwargs):
        model, optimizer, loss, metrics = self.build()

        callbacks = self.build_callbacks(save_dir=save_dir)
        model.fit(
            x,
            y,
            # validation_split=0.2,
            validation_data=(valid_x, valid_y),
            epochs=1000,
            batch_size=32,
            callbacks=callbacks,
        )


def get_word_index_and_label_index(filepath):
    word_index, label_index = {'<pad>': 0}, {'<pad>': 0}
    for line in read_tsv_as_sentence(file_path=filepath):
        for word, label in line:
            word_index.setdefault(word, len(word_index))
            label_index.setdefault(label, len(label_index))
    return word_index, label_index


def get_input_datas(filepath, word_indices, label_indices, max_length=128):
    x_inputs, y_inputs = [], []
    for line in read_tsv_as_sentence(file_path=filepath):
        x_input, y_input = [], []
        for word, label in line:
            word_index = word_indices[word]
            label_index = label_indices[label]
            x_input.append(word_index)
            y_input.append(label_index)
        x_inputs.append(x_input)
        y_inputs.append(y_input)
    return (
        tf.keras.preprocessing.sequence.pad_sequences(
            x_inputs, maxlen=max_length, padding='post',
            truncating='post', value=word_indices['<pad>']
        ),
        tf.keras.preprocessing.sequence.pad_sequences(
            y_inputs, maxlen=max_length, padding='post',
            truncating='post', value=label_indices['<pad>']
        )
    )


def load_indices():
    # train_word_indices, train_label_indices = get_word_index_and_label_index(get_resource(ctb5.CTB5_POS_TRAIN))
    # test_word_indices, test_label_indices = get_word_index_and_label_index(get_resource(ctb5.CTB5_POS_TEST))
    # dev_word_indices, dev_label_indices = get_word_index_and_label_index(get_resource(ctb5.CTB5_POS_DEV))
    # all_word_indices = {**dev_word_indices, **test_word_indices, **train_word_indices}
    # all_label_indices = {**dev_label_indices, **test_label_indices, **train_label_indices}
    # with open("all_word_indices", "w") as f:
    #     f.write(json.dumps(dict(sorted(all_word_indices.items(), key=lambda x: x[1]))))
    # with open("all_label_indices", "w") as f:
    #     f.write(json.dumps(dict(sorted(all_label_indices.items(), key=lambda x: x[1]))))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_word_indices"), "r") as f:
        all_word_indices = json.loads(f.read())
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_label_indices"), "r") as f:
        all_label_indices = json.loads(f.read())
    return all_word_indices, all_label_indices


def train():
    word_indices, label_indices = load_indices()
    path = "/home/yuzhang/.hanlp/thirdparty/dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh/wiki.zh.bin"

    train_file = get_resource(ctb5.CTB5_POS_TRAIN)

    r_model = RNNPartOfSpeechTaggerTF(model_path=path, word_indices=word_indices, units=len(label_indices))

    x_train_inputs, y_train_inputs = get_input_datas(filepath=train_file, word_indices=word_indices,
                                                     label_indices=label_indices)
    x_test_inputs, y_test_inputs = get_input_datas(filepath=get_resource(ctb5.CTB5_POS_TEST), word_indices=word_indices,
                                                   label_indices=label_indices)
    r_model.fit(x=x_train_inputs, y=y_train_inputs, valid_x=x_test_inputs, valid_y=y_test_inputs,
                save_dir='/tmp/fast_pos/')

    r_model.model.evaluate(x_test_inputs, y_test_inputs)


def evaluate():
    path = "/home/yuzhang/.hanlp/thirdparty/dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh/wiki.zh.bin"

    word_indices, label_indices = load_indices()
    x_dev_inputs, y_dev_inputs = get_input_datas(
        filepath=get_resource(ctb5.CTB5_POS_DEV),
        word_indices=word_indices,
        label_indices=label_indices
    )

    r_model = RNNPartOfSpeechTaggerTF(model_path=path, word_indices=word_indices, units=len(label_indices))
    r_model.build()
    r_model.load_weights(save_dir='/tmp/fast_pos')

    r_model.model.evaluate(x_dev_inputs, y_dev_inputs)


def predict():
    path = "/home/yuzhang/.hanlp/thirdparty/dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh/wiki.zh.bin"
    word_indices, label_indices = load_indices()

    r_model = RNNPartOfSpeechTaggerTF(model_path=path, word_indices=word_indices, units=len(label_indices))
    r_model.build()
    r_model.load_weights(save_dir='/tmp/fast_pos')

    x_inputs, y_inputs = get_input_datas(get_resource(ctb5.CTB5_POS_DEV), word_indices, label_indices)
    r_model.model.evaluate(x_inputs, y_inputs)
    y_predicts = r_model.model.predict(x_inputs[100]).argmax(axis=-1).flatten()
    print(y_inputs[100], y_predicts)


if __name__ == '__main__':
    # train()
    # evaluate()
    predict()
