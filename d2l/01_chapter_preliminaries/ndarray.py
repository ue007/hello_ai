##################################################################################################################
# 深度学习存储和操作数据的主要接口是张量（维数组）。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换其他Python对象。
##################################################################################################################

# https://pytorch.org/docs/stable/torch.html
import torch

print(torch.__version__)

# https://devdocs.io/pytorch~2/generated/torch.arange
# arange 创建一个行向量 x
x = torch.arange(12)
print(x)
print(x.shape)

# Returns the total number of elements in the input tensor.
print(x.numel()) 

# a = torch.randn(1, 2, 3, 4, 5)
# print(torch.numel(a)) 

# Returns a tensor with the same data and number of elements as input, but with the specified shape.
X = x.reshape(3, 4)
print(X)

print(torch.zeros((3, 3, 4)))

print(torch.randn(3, 4))

# 运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y) 
print(x * y)
print(x / y) 
print(x ** y)  # **运算符是求幂运算

# https://devdocs.io/pytorch~2/generated/torch.exp
# Returns a new tensor with the exponential of the elements of the input tensor input.
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
print(X)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)

# Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
print( torch.cat((X, Y), dim=0))
print( torch.cat((X, Y), dim=1))

print(X == Y)

## 广播机制: 即使形状不同，我们仍然可以通过调用*广播机制*（broadcasting mechanism）来执行按元素操作
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)

# 索引和切片  [-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素]
print(X)
print(X[-1])
print(X[1:3])

X[1, 2] = 99
print(X)

# 节省内存
before = id(Y)
Y = Y + X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# 转换为其他Python对象
A = X.numpy()
B = torch.tensor(A)
# https://devdocs.io/numpy~2.0/
print(type(A))
# https://devdocs.io/pytorch~2/tensors
print(type(B))

# 将大小为1的张量转换为Python标量
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))