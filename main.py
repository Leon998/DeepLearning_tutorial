import torch
import numpy as np
from torch.nn import functional as F


X = torch.arange(10)
print(X)
X = X.reshape((2, 5))
print(X)
inputs = F.one_hot(X.T, 12)
# print(inputs)
print(inputs.shape)
outputs = []
for X in inputs:
    Y = X
    outputs.append(Y)

print(outputs)
print(torch.cat(outputs, dim=0))
print(torch.cat(outputs, dim=0).shape)

