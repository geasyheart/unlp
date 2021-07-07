# -*- coding: utf8 -*-

#
import torch
from alnlp.modules.util import lengths_to_mask
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TestDataset(Dataset):
    def __init__(self):
        self.batch_size = 2
        self.samples = [
            ['你好', '中国'],
            ['我', '爱', '你'],
            ['我', '在', '这里', '吃饭']
        ]
        self.tokenizer = AutoTokenizer.from_pretrained('ckiplab/albert-tiny-chinese')

    def __getitem__(self, idx):
        m = {}
        sample = self.samples[idx]
        f = self.tokenizer.batch_encode_plus(sample, add_special_tokens=False)
        input_ids = []
        [input_ids.extend(_) for _ in f['input_ids']]
        input_ids = [self.tokenizer.cls_token_id] + input_ids[:126] + [self.tokenizer.sep_token_id]
        padding_size = 128 - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_size
        m['input_ids'] = torch.tensor(input_ids)
        m['sent_lens'] = len(sample)
        m['words'] = sample

        return m

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        batches = {key: [] for key in batch[0].keys()}

        for _batch in batch:
            for key, value in _batch.items():
                batches[key].append(value)
        batches['input_ids'] = pad_sequence(batches['input_ids'], batch_first=True)
        batches['sent_lens'] = torch.tensor(batches['sent_lens'])
        batches['mask'] = lengths_to_mask(batches['sent_lens'])
        return batches

    def loader(self):
        return DataLoader(self, collate_fn=self.collate_fn, batch_size=self.batch_size)


if __name__ == '__main__':
    ds = TestDataset()
    for i in ds.loader():
        print(i)
