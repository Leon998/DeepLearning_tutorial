# 本模块更符合以后在大型应用中的操作流程
# 重点掌握！！！

import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l


# ========================生成数据================================ #
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 产生数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 80)
# 产生小批量
batch_size = 8
data_iter = load_array((features, labels), batch_size)
# 测试小批量
X, y = next(iter(data_iter))
print("Test of a small batch:")
print(type((X, y)))  # 可以看到，DataLoader后返回的数据类型是元组
print(X, '\n', y)


# =========================定义模型=============================== #
# 在PyTorch中，全连接层在Linear类中定义。 值得注意的是，我们将两个参数传递到nn.Linear中。
# （2，1）表示的是w,b的维度
# 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2, 1))
print('Net structure is: ', net)
# 初始化模型参数
# 在使用net之前，我们需要初始化模型参数。 如在线性回归模型中的权重和偏置。
# 深度学习框架通常有预定义的方法来初始化参数。
# 在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
# 计算均方误差使用的是MSELoss类，也称为平方 L2 范数。 默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# ========================训练================================= #
num_epochs = 5
for epoch in range(num_epochs):
    for step, (X, y) in enumerate(data_iter):
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        l_in = loss(net(features), labels)
        print('Epoch:{} | step:{} | loss:{} '.format(epoch, step, l_in))
    l = loss(net(features), labels)
    print(f'epoch {epoch}, temploss {l:f}')

# ========================测试============================ #
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
