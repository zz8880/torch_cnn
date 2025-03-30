import torch
from numpy.matlib import randn

print(torch.ones(2, 3))
print(torch.zeros(3, 3))
print(torch.rand(3, 4))
print(torch.randint(0, 10, (2,3)))
#生成符合正态分布的数据
print(randn(3, 4))
a = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])
print(a)
b = torch.rand_like(a, dtype=float)
print(b)
c = b.view(6)
print(c)
d = b.view(2,3)
print(d)
print(c[1])
print(c[1].item())