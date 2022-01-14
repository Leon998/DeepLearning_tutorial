# class MyNumbers:
#     def __iter__(self):
#         self.a = 1
#         return self
#
#     def __next__(self):
#         x = self.a
#         self.a += 2
#         return x
#
#
# myclass = MyNumbers()
# myiter = iter(myclass)
# print(myiter)
#
# for i in range(5):
#     l = next(myiter)
#     print(l)

import torch

a = torch.arange(1, 10).reshape(3, 3)
b = torch.arange(11, 20).reshape(3, 3)
c = torch.arange(21, 30).reshape(3, 3)
print(a)
print('Dimension_0 is like:', a[:, 0])
print('Dimension_1 is like:', a[0, :])

d = torch.stack((a, b, c), dim=2)
print(d)
print(d.shape)
print('Dimension_0 of d is like:', d[:, 0, 0])
print('Dimension_1 of d is like:', d[0, :, 0])
print('Dimension_2 of d is like:', d[0, 0, :])

e = torch.stack((d, 2*d), dim=3)
print(e.shape)
print('Dimension_0 of e is like:', e[:, 0, 0, 0])
print('Dimension_1 of e is like:', e[0, :, 0, 0])
print('Dimension_2 of e is like:', e[0, 0, :, 0])
print('Dimension_3 of e is like:', e[0, 0, 0, :])