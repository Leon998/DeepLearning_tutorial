import torch
from torch import nn

# ==============================参数访问================================== #
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

# print('The output tensor: ',net(X))
# # 访问目标参数
# print('The second linear parameters: ',net[2].state_dict())
# print('The type of bias: ',type(net[2].bias))
# print('The bias content: ',net[2].bias)
# print('The bias data: ',net[2].bias.data)
#
# # 一次性访问所有参数
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet)
# 因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。
# 下面，我们访问第一个主要的块中、第二个子块的第一层的偏置项。
# print(rgnet[0][1][0].bias.data)


# =================================参数初始化================================ #
# 内置初始化
def init_normal(m):
    """
    调用内置的初始化器，将所有权重参数初始化为标准差为0.01的高斯随机变量，且将偏置参数设置为0。
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
var = net[0].weight.data, net[0].bias.data
print(var)

# 还可以对某些块应用不同的初始化方法。
# 例如，下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
var = net[0].weight[:2]
print(var)
