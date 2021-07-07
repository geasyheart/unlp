# -*- coding: utf8 -*-


#

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co


class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = self.data.shape[0]

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return self.size


if __name__ == '__main__':
    data = np.array([[1, 2], [3, 4], [5, 6]])
    loader = DataLoader(MyDataSet(data), batch_size=2, shuffle=True)
    for batch_data in loader:
        print(batch_data)
