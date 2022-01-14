import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

torch.manual_seed(777)  # reproducibility
#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]  # hihell
one_hot_lookup = [[1, 0, 0, 0, 0],  # 0
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]]  # 4

y_data = [1, 0, 2, 3, 3, 4]  # ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]
print(x_one_hot)
inputs = Variable(torch.Tensor(x_one_hot))
print(inputs.dtype, inputs)
labels = Variable(torch.LongTensor(y_data))
print(labels.dtype,labels)
num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1  # one sentence
sequence_length = 1  # One by one
num_layers = 1  # one-layer rnn


class Model(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True)

    def forward(self, hidden, x):
        # Reshape input (batch first)
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)  # out:torch.Size([1, 1, 5])
        return hidden, out.view(-1, num_classes)  # return 的out的大小torch.Size([1, 5])

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float32))


# Instantiate RNN model
model = Model(input_size, hidden_size)
# print(model)
# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
optimizer.zero_grad()
hidden = model.init_hidden()
print(hidden.dtype, hidden)

# ===============================Training================================ #
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()

    for input, label in zip(inputs, labels):
        # print(input.size(), label.size())
        label = label.view(1)
        hidden, output = model(hidden, input)
        val, idx = output.max(1)  # 找到output里最大的值和他的索引，这就是预测的字符，然后再把他的值和label做crossentropy
        loss += loss_func(output, label)

    if epoch % 20 == 0:
        print(", epoch: %d, loss: %1.3f" % (epoch, loss.item()))

    loss.backward()
    optimizer.step()

# Prediction
hidden = model.init_hidden()
print("predicted string: ")
for input, label in zip(inputs, labels):
    # print(input.size(), label.size())
    label = label.view(1)
    hidden, output = model(hidden, input)
    val, idx = output.max(1)  # 找到output里最大的值和他的索引，这就是预测的字符，然后再把他的值和label做crossentropy
    sys.stdout.write(idx2char[idx.data[0]])
