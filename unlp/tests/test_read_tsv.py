# -*- coding: utf8 -*-

#
import tensorflow as tf

from unlp.layers.transformers.loader_tf import build_transformer

_, tokenizer = build_transformer(num_labels=1, transformer='albert_base_zh')


def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        select_columns=['text_a', 'label'],
        batch_size=12,
        label_name='label',
        num_epochs=1,
        ignore_errors=True,
        field_delim='\t'

    )
    return dataset


def mapfunc(x, y):
    print(x['text_a'])
    print(y)
    return ""


dataset = get_dataset('/home/yuzhang/.unlp/thirdparty/ernie.bj.bcebos.com/task_data_zh/chnsenticorp/dev.tsv')
dataset.map(map_func=mapfunc)
# for batch in dataset:
#     x, y = batch
#     for line in x['text_a']:
#         # tokenizer.tokenize(line)
#         print(x)
#     break
