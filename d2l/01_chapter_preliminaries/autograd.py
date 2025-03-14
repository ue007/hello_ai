##########################################################################################################################
# 自动微分
##########################################################################################################################

import torch

x = torch.arange(4.0)
print(x) # tensor([0., 1., 2., 3.])

x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)

print(x.grad ) # 默认值是None, 需要一个地方来存储梯度， None

# https://devdocs.io/pytorch~2/generated/torch.dot
# Computes the dot product of two 1D tensors.
y = 2 * torch.dot(x, x)
print(y) # tensor(28., grad_fn=<MulBackward0>)

# https://devdocs.io/pytorch~2/generated/torch.tensor.backward#torch.Tensor.backward
# 通过调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
print(x.grad) # tensor([ 0.,  4.,  8., 12.]) 

# 快速验证这个梯度是否计算正确
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
print(x.grad)
x.grad.zero_()
print(x.grad)

# https://devdocs.io/pytorch~2/generated/torch.sum  
# Returns the sum of all elements in the input tensor.
y = x.sum()
print("x = ", x)
print("y = ", y)
y.backward()
print("x.grad", x.grad)

# torch.sum
a = torch.randn(1, 3)
print(a) # tensor([[ 0.1133, -0.9567,  0.2958]])
print(torch.sum(a)) # tensor(-0.5475)

# 非标量变量的反向传播
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
print("x = ", x)
y = x * x
# 等价于y.backward(torch.ones(len(x)))
print(y.sum().backward())
# print(y.backward())
print(x.grad)

# 分离计算:将某些计算移动到记录的计算图之外
x.grad.zero_()
print("x = ", x)
y = x * x
print("y = ", y)
# https://devdocs.io/pytorch~2/generated/torch.tensor.detach#torch.Tensor.detach
u = y.detach()
print("u = ", u)
z = u * x
print("z = ", z)

# torch.sum(input, *, dtype=None)：返回输入张量input所有元素的和。
z.sum().backward()

print("x.grad = " ,x.grad)

print("u = " ,u)

print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

# Python控制流的梯度计算
# 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度， 分段函数
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d / a)