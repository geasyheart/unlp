# -*- coding: utf8 -*-

#

import torch
from torch import nn

loss = nn.CrossEntropyLoss()

# inputs = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(inputs, target)
# print(output)
# output.backward()
#

inputs = torch.randn(32, 128, 5, requires_grad=True)
target = torch.empty(32, 128, dtype=torch.int64).random_(5)
rs = loss(inputs.view(-1, 5), target.view(-1))
print(rs)
rs.backward()
