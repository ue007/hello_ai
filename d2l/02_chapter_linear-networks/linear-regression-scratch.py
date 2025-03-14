# 线性回归的从零开始实现
# 在了解线性回归的关键思想之后，我们可以开始通过代码来动手实现线性回归了。 在这一节中，(我们将从零开始实现整个方法， 包括数据流水线、模型、损失函数
# 和小批量随机梯度下降优化器)。 虽然现代的深度学习框架几乎可以自动化地进行所有这些工作，但从零开始实现可以确保我们真正知道自己在做什么。 
# 同时，了解更细致的工作原理将方便我们自定义模型、自定义层或自定义损失函数。 在这一节中，我们将只使用张量和自动求导。 在之后的章节中，
# 我们会充分利用深度学习框架的优势，介绍更简洁的实现方式。
# ##########################################################################################################################

import random
import torch
from d2l import torch as d2l

# 1. 生成数据集： 根据带有噪声的线性模型构造一个人造数据集。
def synthetic_data(w, b, number):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (number, len(w)))
    print("X = ", X)
    y = torch.matmul(X, w) + b
    print("y = ", y)
    y += torch.normal(0, 0.01, y.shape)
    print("y = ", y)
    return X, y.reshape((-1, 1))
 
true_w = torch.tensor([2, -3.4])
print(true_w)

true_b = 4.2
print(true_b)

features, labels = synthetic_data(true_w, true_b, 10)
print(features)
print(labels)
print('features:', features[0],'\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)

# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
# 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')