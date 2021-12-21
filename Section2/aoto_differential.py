import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
y = x * x
print(y)
y.sum().backward()
print(x.grad)  # 默认值是None