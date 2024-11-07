import torch
from torch import autograd
# hello https://blog.csdn.net/2303_77275067/article/details/142665623
# 假设我们有一个简单的操作
x = torch.tensor(2.0, requires_grad=True)  # 创建一个张量 x
y = x ** 2  # y = x^2
 
# 计算 y 相对于 x 的梯度
y.backward()  # 计算梯度
print(x.grad)  # 输出: 4.0，因为 dy/dx = 2x，x=2 时，dy/dx=4

# example
x = torch.tensor(1.)
print(x)

a = torch.tensor(1., requires_grad=True)
print(a)

b = torch.tensor(2., requires_grad=True)
print(b)

c = torch.tensor(3., requires_grad=True)
print(c)

y = a**2 * x + b * x + c
# y = a * x **2 + b * x + c

print('before', a.grad, b.grad, c.grad)

# torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)
grads = autograd.grad(y, [a, b, c])

print('after:', grads[0], grads[1], grads[2])
