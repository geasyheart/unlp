# -*- coding: utf8 -*-

#
from typing import List

import pandas as pd
import tensorflow as tf

from unlp.common.keras_component import KerasComponent
from unlp.common.transform_tf import TransformTF
from unlp.datasets.classification.sentiment import CHNSENTICORP_ERNIE_TRAIN, CHNSENTICORP_ERNIE_DEV
from unlp.layers.transformers.loader_tf import build_transformer
from unlp.utils.io_util import get_resource


class TransformerClassifierTF(KerasComponent):
    def __init__(self):
        super(TransformerClassifierTF, self).__init__()
        self.tokenizer = None

    def build_model(self, transformer):
        model, self.tokenizer = build_transformer(
            transformer=transformer, max_seq_len=128,
            num_labels=1, tagging=False
        )
        return model

    def build_loss(self, loss=None):
        # return tf.keras.losses.binary_crossentropy
        return tf.keras.losses.BinaryCrossentropy()

    def build_metrics(self, metrics=None):
        # return [
        #     tf.keras.metrics.binary_accuracy
        # ]
        return [
            tf.keras.metrics.BinaryAccuracy()
        ]

    def predict(self, x: List):
        result = []
        seq = TransformTF(pd.DataFrame({"text_a": x}), predict=True, transformer='albert_base_zh')
        for x in seq:
            y_predicts = self.model.predict_on_batch(x)

            for y_predict in y_predicts:
                y_to_flat = y_predict[0]
                if y_to_flat >= 0.5:
                    result.append(1)
                else:
                    result.append(0)
        return result


def train():
    train_data = get_resource(CHNSENTICORP_ERNIE_TRAIN)
    dev_data = get_resource(CHNSENTICORP_ERNIE_DEV)

    train_df = pd.read_csv(train_data, sep='\t')
    dev_df = pd.read_csv(dev_data, sep='\t')

    tct = TransformerClassifierTF()
    tct.fit(
        train_data=TransformTF(train_df, "albert_base_zh"),
        dev_data=TransformTF(dev_df, "albert_base_zh", shuffle=False),
        save_dir="/tmp/classifier",
        transformer="albert_base_zh"
    )


if __name__ == '__main__':
    train()
    # tct = TransformerClassifierTF()
    # tct.build(transformer='albert_base_zh')
    # tct.load_weights(save_dir="/tmp/classifier", filename='model.1619753457.h5')

    # evaluate
    # test_data = get_resource(CHNSENTICORP_ERNIE_TEST)
    #
    # test_df = pd.read_csv(test_data, sep='\t')
    # result = tct.model.evaluate(TransformTF(test_df, "albert_base_zh", shuffle=False))
    # print(result)

    # predict
    # for x in TransformTF(dev_df, "albert_base_zh", shuffle=False):
    #     print(tct.model.predict_on_batch(x))
    result = tct.predict(
        ['今天酒店卫生很干净', ' 前台接待太差，酒店有A B楼之分，本人check－in后，前台未告诉B楼在何处，并且B楼无明显指示；房间太小，根本不像4星级设施，下次不会再选择入住此店啦。'])
    print(result)
