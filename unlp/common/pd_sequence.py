# -*- coding: utf8 -*-

#

import math

import numpy as np
import pandas as pd
import tensorflow as tf


class DataFrameSequence(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, batch_size: int = 32, shuffle: bool = True,
                 max_length: int = 128):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length

        self.file_size = len(self.df)
        self.indexes = np.arange(self.file_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]

        return batch_df

    def __len__(self):
        return math.ceil(self.file_size / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(333).shuffle(self.indexes)


if __name__ == '__main__':
    tsv_df = pd.DataFrame({"names": ["A", 'B', 'C', 'D', 'E']})
    tsv = DataFrameSequence(tsv_df, batch_size=2)

    for batch_df in tsv:
        print(batch_df)
