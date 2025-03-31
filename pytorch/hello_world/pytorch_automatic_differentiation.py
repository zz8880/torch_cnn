import torch
x = torch.ones((2, 2), requires_grad=True)
print("x = ", x)

y = x + 2
print("y=", y)

z = y * y * 3
print("z = ", z)

out = z.mean()
print("out = ", out)

out.backward()
print(x.grad)