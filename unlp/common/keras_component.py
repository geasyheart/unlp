# -*- coding: utf8 -*-

#
import os
from abc import ABCMeta
from typing import List, Any

import tensorflow as tf


class KerasComponent(object, metaclass=ABCMeta):
    def __init__(self):
        self.model: tf.keras.Model

    def build_model(self, *args, **kwargs) -> tf.keras.Model:
        raise NotImplementedError

    def build_loss(self, loss=None):
        return tf.keras.losses.SparseCategoricalCrossentropy()
        # return "sparse_categorical_crossentropy"

    def build_optimizer(self, optimizer=None):
        if optimizer:
            return optimizer
        return tf.keras.optimizers.Adam()

    def build_metrics(self, metrics=None) -> List[Any]:

        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        return [metric]
        # return ["accuracy"]

    @staticmethod
    def step_decay(epoch, initial_lrate=0.00001):
        # if epoch < 10:
        #     lr = 2e-4
        # elif 10 <= epoch < 30:
        #     lr = 2e-5
        # else:
        #     lr = 2e-6
        # return lr
        if epoch < 3:
            lr = 1e-5
        else:
            lr = 1e-6
        return lr

    def build_callbacks(self, save_dir: str) -> List[Any]:
        model_save_dir = os.path.join(save_dir, "weights")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        log_save_dir = os.path.join(save_dir, 'logs')
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)

        return [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_save_dir, 'model.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_save_dir,
            ),
            # 动态调节学习率
            tf.keras.callbacks.LearningRateScheduler(self.step_decay, verbose=2),

            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                verbose=1,
                patience=3
            )
        ]

    def build(self, transformer):
        model = self.build_model(transformer=transformer)
        optimizer = self.build_optimizer()
        loss = self.build_loss()
        metrics = self.build_metrics()
        # if not self.model.built:
        #     pass
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model = model
        return model, optimizer, loss, metrics

    def fit(self, train_data, dev_data, save_dir, transformer: str, epochs: int = 1000, batch_size=32):
        model, optimizer, loss, metrics = self.build(transformer=transformer)
        model.summary()
        callbacks = self.build_callbacks(save_dir=save_dir)

        model.fit(
            train_data,
            validation_data=dev_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

    def load_weights(self, save_dir, filename='model.h5'):
        assert self.model.built or self.model.weights, 'model need build first.'
        if os.path.isfile(save_dir):
            self.model.load_weights(save_dir)
            return
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            filepath = os.path.join(save_dir, "weights", filename)
        self.model.load_weights(filepath)
