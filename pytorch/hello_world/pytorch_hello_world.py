import torch
a = torch.tensor([1, 2, 3], dtype = int)
print(a)
print(a.dtype)

b = torch.tensor([4, 5, 6], dtype=float)
print(b)

tensor = torch.tensor([[1,2,3],
                      [4,5,6]])
#数据的维度
print(tensor.ndim)
#数据的形状
print(tensor.shape)
print(tensor.size())
print(tensor.dtype)