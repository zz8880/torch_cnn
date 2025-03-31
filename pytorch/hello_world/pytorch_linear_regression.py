import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable, variable
import torch

x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise
#y_data = x_data * 0.1 + 0.2

plt.scatter(x_data, y_data)
#plt.show()

x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

#把numpy数据变成tensor
x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)
inputs = Variable(x_data)
target = Variable(y_data)

#构建神经网络
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

#定义模型
model = LinearRegression()
#定义代价函数
mes_loss = nn.MSELoss()
#定义优化器
optimizer = optim.SGD(model.parameters(), lr= 0.1)

for name, parameters in model.named_parameters():
    print('name:{}, param:{}'.format(name, parameters))