import torch
from torch import nn


X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

# multiple tunnels
X1 = torch.cat((X, X + 1), 1)
print(X1)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X1))