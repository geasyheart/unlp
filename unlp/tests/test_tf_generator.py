# -*- coding: utf8 -*-

#

import numpy as np
import tensorflow as tf


def generator(batch_size=30):
    for i in range(100):
        yield np.array([101, i, 102])


dataset = tf.data.Dataset.from_generator(generator=generator, args=(10,), output_shapes=(3,),
                                         output_types=tf.int32)

if __name__ == '__main__':
    # for batch in generator():
    #     print(batch.shape)
    for batch in dataset.batch(batch_size=10).take(10):
        print('here'.center(60, '-'))
        print(batch.shape)
        print(batch.numpy())
