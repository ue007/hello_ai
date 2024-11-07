
import torch
 
# 创建一个标量（0维张量）
scalar = torch.tensor(5)
 
# 创建一个向量（1维张量）
vector = torch.tensor([1, 2, 3])
 
# 创建一个矩阵（2维张量）
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
 
# 创建一个三维张量
three_d_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
 
print(scalar)
print(vector)
print(matrix)
print(three_d_tensor)