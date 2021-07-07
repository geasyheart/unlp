# -*- coding: utf8 -*-

#

import numpy as np
import pandas as pd

from unlp.common.pd_sequence import DataFrameSequence
from unlp.common.tokenizer import Tokenizer
from unlp.layers.transformers.bert_tokenizer import get_tokenizer


class TransformTF(DataFrameSequence):
    def __init__(self, df: pd.DataFrame, transformer=None, batch_size: int = 32, shuffle: bool = True,
                 max_length: int = 128, predict=False):
        super(TransformTF, self).__init__(
            df=df,
            batch_size=batch_size,
            shuffle=shuffle,
            max_length=max_length
        )
        self.predict = predict

        self.tokenizer = get_tokenizer(transformer=transformer)
        self.t = Tokenizer(self.tokenizer.vocab)

    def __getitem__(self, index):
        batch_df = super(TransformTF, self).__getitem__(index=index)

        input_ids_lst, token_type_ids_lst = [], []
        labels = []

        for index, row in batch_df.iterrows():
            label: int = row.get('label')
            labels.append(label)

            text: str = row['text_a']
            input_ids, token_type_ids = self.t.encode(first=text, max_len=self.max_length)
            input_ids_lst.append(input_ids)
            token_type_ids_lst.append(token_type_ids)

        if not self.predict:
            return [
                       np.array(input_ids_lst, dtype=np.int32),
                       np.array(token_type_ids_lst, dtype=np.int32)
                   ], np.array(labels, dtype=np.int32)
        return np.array(input_ids_lst, dtype=np.int32), np.array(token_type_ids_lst, dtype=np.int32)


if __name__ == '__main__':
    # tsv_df = pd.DataFrame({"names": ["A", 'B', 'C', 'D', 'E']})
    # tsv = TransformTF(tsv_df, batch_size=2)
    dev_df = pd.read_csv('/home/yuzhang/.unlp/thirdparty/ernie.bj.bcebos.com/task_data_zh/chnsenticorp/dev.tsv',
                         sep='\t')
    for batch_df in TransformTF(dev_df, transformer='albert_base_zh'):
        print(batch_df)
