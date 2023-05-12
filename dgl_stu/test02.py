import torch

m = torch.nn.ELU()
input = torch.randn(2)
output = m(input)

print(input)
print(output)