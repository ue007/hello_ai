##########################################################################################################################
# 线性代数: 本节将介绍线性代数中的基本数学对象、算术和运算，并用数学符号和相应的代码实现来表示它们。
##########################################################################################################################
import torch

# 标量:标量由只有一个元素的张量表示
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x**y)

# 向量可以被视为标量值组成的列表
x = torch.arange(4)
print(x)

# 长度、维度和形状
print(len(x))
print(x.shape)

# 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
# 矩阵的转置
print(A.T)

# 对称矩阵（symmetric matrix）
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)

# 张量: 就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 张量算法的基本性质
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, A + B)

# 两个矩阵的按元素乘法称为*Hadamard积*（Hadamard product）（数学符号$\odot$）
print(A * B)

# 降维:我们可以对任意张量进行的一个有用的操作是[计算其元素的和]
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())

# 表示任意形状张量的元素和
print(A.shape, A.sum())

# 指定张量沿哪一个轴来通过求和降低维度
A_sum_axis0 = A.sum(axis=0)
print("A = ", A)
print("axis=0")
print(A_sum_axis0, A_sum_axis0.shape)

# 指定axis=1将通过汇总所有列的元素降维（轴1）。因此，输入轴1的维数在输出形状中消失。
A_sum_axis1 = A.sum(axis=1)
print("axis=1")
print(A_sum_axis1, A_sum_axis1.shape)

# 一个与求和相关的量是平均值（mean或average）
print(A.mean(), A.sum() / A.numel())

# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print("sum_A",sum_A)

# 点积（Dot Product）
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

# 矩阵-向量积
A.shape, x.shape, torch.mv(A, x)

# 矩阵-矩阵乘法
B = torch.ones(4, 3)
torch.mm(A, B)

# 范数
u = torch.tensor([3.0, -4.0])
torch.norm(u)