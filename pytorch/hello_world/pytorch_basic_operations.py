import torch
a = torch.randint(1, 5, (2, 3))
b = torch.randint(1, 5, (2, 3))
print(b)
print(b)

print(a + b)
print(torch.add(a, b))

result = torch.zeros(3, 2)
torch.add(a, b, out = result)
print(result)

# a = a + b
a.add_(b)

a - b
a * b
a / b

#取余数
a % b

#a/b 后取整
a // b

tensor = torch.ones(3, 5)
a = a.float()

print("--------")

print(a)
print(tensor)

#矩阵乘法
print(torch.matmul(a, tensor))

print(torch.sum(tensor))

print(torch.min(a))
#最小值所在的位置
print(torch.argmin(a))

print(torch.max(b))
#最大值所在的位置
print(torch.argmax(a))

#平均值
torch.mean(tensor)