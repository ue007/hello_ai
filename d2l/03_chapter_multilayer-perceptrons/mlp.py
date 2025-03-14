##########################################################################################################################
# 多层感知机
##########################################################################################################################

# 隐藏层

# 线性模型可能会出错

# 在网络中加入隐藏层

# 从线性到非线性

# 通用近似定理

# 激活函数

# import torch
# from d2l import torch as d2l

# # ReLU函数 最受欢迎的激活函数是修正线性单元（Rectified linear unit，ReLU）
# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

# # sigmoid函数
# y = torch.sigmoid(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# # 清除以前的梯度
# x.grad.data.zero_()
# y.backward(torch.ones_like(x),retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))


# # tanh函数
# y = torch.tanh(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

# # 清除以前的梯度
# x.grad.data.zero_()
# y.backward(torch.ones_like(x),retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

import torch
from d2l import torch as d2l

# ReLU函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

# sigmoid函数
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

# tanh函数
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

print(x,y)