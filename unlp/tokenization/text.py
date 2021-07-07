# -*- coding: utf8 -*-

#
from linecache import getline

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import BertTokenizerFast

from unlp.utils.span_util import bmes_of

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextTokenizationDataset(Dataset):
    def __init__(self, text_file):
        self.text_file = text_file
        self.size = self.get_line_number()

    def get_line_number(self):
        size = 0
        with open(self.text_file, encoding='utf-8') as f:
            for _ in f:
                size += 1
        return size

    def __getitem__(self, index) -> T_co:
        """因为index从0开始,getline从正常1开始，所以此处加1"""
        return getline(filename=self.text_file, lineno=index + 1).strip()

    def __len__(self):
        return self.size


class BMESTokenizationDataset(TextTokenizationDataset):
    def __init__(self, text_file, tokenizer, max_len=128):
        super(BMESTokenizationDataset, self).__init__(text_file=text_file)
        self.tokenizer: BertTokenizerFast = tokenizer
        self.max_len = max_len

        self.label_map = {'B': 1, 'M': 2, 'E': 3, 'S': 4}

    def __getitem__(self, index):
        line = super(BMESTokenizationDataset, self).__getitem__(index=index)
        tokens, labels = bmes_of(sentence=line, segmented=True)

        token_type_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [self.label_map[label] for label in labels]

        token_type_ids = [self.tokenizer.cls_token_id] + token_type_ids[:self.max_len - 2] + [
            self.tokenizer.sep_token_id]
        label_ids = [self.tokenizer.pad_token_id] + label_ids[:self.max_len - 2] + [self.tokenizer.pad_token_id]
        pad_tokens = (self.max_len - len(token_type_ids)) * [self.tokenizer.pad_token_id]

        return [
                   torch.tensor(token_type_ids + pad_tokens, dtype=torch.long, device=device),
                   torch.zeros(self.max_len, dtype=torch.long, device=device)
               ], torch.tensor(label_ids + pad_tokens, dtype=torch.long, device=device)
