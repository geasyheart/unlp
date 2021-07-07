# -*- coding: utf8 -*-

#
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


def encoder_texts(texts, tokenizer):
    """
    每个text in texts 的长度都是固定的

    比如:

    我     在    中国
    [200] [202] [203, 204]
    按照每个词获取embedding.
    Args:
        texts:
        tokenizer:

    Returns:

    """
    pad_index = tokenizer.pad_token_id
    max_words_len = max([max([len(word) for word in text[1:]]) for text in texts])

    texts_matrix = []
    for text in texts:
        text_matrix = []

        tmp = tokenizer.batch_encode_plus(text, add_special_tokens=False)
        for input_ids in tmp['input_ids']:
            pad_input_ids = input_ids + (max_words_len - len(input_ids)) * [pad_index]
            text_matrix.append(pad_input_ids)
        texts_matrix.append(text_matrix)

    max_sequence_len = max([len(i) for i in texts_matrix])
    pad_lens = []
    for text_matrix in texts_matrix:
        if len(text_matrix) < max_sequence_len:
            pad_len = max_sequence_len - len(text_matrix)
            text_matrix.extend([[pad_index] * max_words_len] * pad_len)
            pad_lens.append(pad_len)
        else:
            pad_lens.append(0)
    return torch.tensor(texts_matrix), pad_lens


class DependencyDataset(Dataset):
    def __init__(self, file, tokenizer, batch_size=32):
        self.file = file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.cls_as_bos = True

        if file.endswith('.conllu'):
            # See https://universaldependencies.org/format.html
            field_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS',
                           'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
        else:
            field_names = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                           'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
        self.field_names = field_names

        self.data = [i for i in self.read_conllx(file=file)]
        self.size = len(self.data)

        self.rel_ids = self.get_rel_ids()

    def read_conllx(self, file):
        sent = {k: [] for k in self.field_names}
        with open(file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    yield sent
                    sent = {k: [] for k in self.field_names}
                else:
                    cells = line.split('\t')
                    for key, value in zip(self.field_names, cells):
                        if key in ('ID', 'HEAD'):
                            value = int(value)
                        sent[key].append(value)
        if sent['ID']:
            yield sent

    def get_rel_ids(self):
        rel_ids = {}
        for i in self.data:
            rels = i['DEPREL']
            for rel in rels:
                id_ = rel_ids.get(rel)
                if id_ is None:
                    id_ = len(rel_ids)
                    rel_ids[rel] = id_
        with open('rel_ids.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(rel_ids, ensure_ascii=False, indent=4))
        return rel_ids

    def get_rel_id(self, rels):
        return [self.rel_ids[rel] for rel in rels]

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return self.size

    def my_coolate_fn(self, batch):
        texts, heads, rels = [], [], []
        for i in batch:
            # <ROOT> -> cls
            texts.append(['[CLS]', *i['FORM']])
            heads.append([0, *i['HEAD']])
            rels.append([0, *self.get_rel_id(i['DEPREL'])])
        words, texts_pad_lens = encoder_texts(texts=texts, tokenizer=self.tokenizer)
        for index, (length, head, rel) in enumerate(zip(texts_pad_lens, heads, rels)):
            heads[index] = head + [0] * length
            rels[index] = rel + [0] * length
        return texts, words, torch.tensor(heads), torch.tensor(rels)

    def dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, collate_fn=self.my_coolate_fn)


if __name__ == '__main__':
    from unlp.datasets.parsing.ctb8 import CTB8_SD330_DEV
    from unlp.utils.io_util import get_resource
    from transformers import AutoTokenizer

    tokenizers = AutoTokenizer.from_pretrained('ckiplab/albert-tiny-chinese')
    dataset = DependencyDataset(file=get_resource(CTB8_SD330_DEV), tokenizer=tokenizers)
    for d in dataset.dataloader():
        print(d)
    # for d in DataLoader(dataset, batch_size=32, collate_fn=mycollector):
    #     print(d)
