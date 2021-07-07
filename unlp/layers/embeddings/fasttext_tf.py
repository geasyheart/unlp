from typing import Dict

import fasttext
import numpy as np
import tensorflow as tf


def build_fasttext_embedding(model_path: str, words_indices: Dict[str, int]):
    # 获取所有的词，以及词对应的index,此为固定值，不能变
    """
    关于Embedding,返回的就是对应词的vec.

    > hanlp/layers/embeddings/fast_text_tf.py


    举例：
     Input shape:
        2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, input_length, output_dim)`.

    其中的out_dim就是对应word的vec.

    我 是





    :param model_path:
    :param words_indices:
    :return:
    """
    max_words = len(words_indices)
    model = fasttext.load_model(path=model_path)
    embedding_matrix = np.zeros((max_words, model.get_dimension()))
    for word, index in words_indices.items():
        if index == 0:
            assert word == '<pad>'
            continue

        embedding_matrix[index] = model.get_word_vector(word)
    layer = tf.keras.layers.Embedding(
        input_dim=max_words,
        output_dim=model.get_dimension(),
        name='fasttext_embedding',
        # trainable=False,
        weights=[embedding_matrix],
        # mask_zero=True
    )
    return layer


if __name__ == '__main__':
    path = "/home/yuzhang/.hanlp/thirdparty/dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh/wiki.zh.bin"
    word_index = {"你好 ": 1, "Hello": 2}
    build_fasttext_embedding(path, word_index)
