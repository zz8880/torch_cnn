import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable, variable
import torch

x_data = np.linspace(-2, 2, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.2, x_data.shape)
y_data = np.square(x_data) + noise

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
        #初始化 nn.Module
        super(LinearRegression, self).__init__()
        #1-10-1
        self.fc1 = nn.Linear(1, 10)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out

#定义模型
model = LinearRegression()
#定义代价函数
mes_loss = nn.MSELoss()
#定义优化器
optimizer = optim.SGD(model.parameters(), lr= 0.3)

for name, parameters in model.named_parameters():
    print('name:{}, param:{}'.format(name, parameters))

for i in range(2001):
    out = model(inputs)
    #计算loss
    loss = mes_loss(out, target)
    #梯度清零
    optimizer.zero_grad()
    #计算梯度
    loss.backward()
    #修改权值
    optimizer.step()
    if i % 200 == 0:
        print(i, loss.item())

y_pred = model(inputs)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred.data.numpy(), 'r-', lw=3)
plt.show()