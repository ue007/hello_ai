import torch
from torch import nn

# 行,列
input = torch.ones(2,3) 
print(input)

input = torch.ones(5)
print(input)

print(torch.__version__)

print(torch.cuda.is_available())

# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# tensor([1., 1., 1., 1., 1.])
# 2.4.1+cu121
# True 