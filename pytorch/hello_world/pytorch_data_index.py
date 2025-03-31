import torch
tensor = torch.arange(2, 14)
print(tensor)

print(tensor[2])
print(tensor[1:4])
print(tensor[2:-1])
print(tensor[:5])
print(tensor[-3:])
index = [1, 3, 4, 5, 5]
print(tensor[index])
for t in tensor:
    print(t)