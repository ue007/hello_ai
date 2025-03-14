##########################################################################################################################
# 回归（regression）是能为一个或多个自变量与因变量之间关系建模的一类方法。 在自然科学和社会科学领域，回归经常用来表示输入和输出之间的关系。
##########################################################################################################################

# 线性模型

# 损失函数

# 解析解

# 随机梯度下降

# 用模型进行预测 预测（prediction）或推断（inference）。

# 矢量化加速： 在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本。 为了实现这一点，需要(我们对计算进行矢量化， 从而利用线性代数库，而不是在Python中编写开销高昂的for循环)。
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones([n])
b = torch.ones([n])

# 我们定义一个计时器
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

# 我们使用for循环，每次执行一位的加法
c = torch.zeros(n)

timer = Timer()

for i in range(n):
    c[i] = a[i] + b[i]
    
# 格式化输出计时结果,f-string 是 Python 3.6 及以上版本引入的一种字符串格式化机制，它允许在字符串中嵌入表达式，只需要在字符串前加上 f 或 F 前缀，然后将表达式放在花括号 {} 内。
formatted_time = f'{timer.stop():.5f} sec'
print(formatted_time) #0.13534 sec

timer.start()
d = a + b
formatted_time = f'{timer.stop():.5f} sec'
print(formatted_time) #0.00002 sec

'''
结果很明显，第二种方法比第一种方法快得多。 矢量化代码通常会带来数量级的加速。 另外，我们将更多的数学运算放到库中，而无须自己编写那么多的计算，从而减少了出错的可能性。
'''

# 正态分布与平方损失
# 我们定义一个Python函数来计算正态分布
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',ylabel='p(x)', figsize=(4.5, 2.5),legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


# 从线性回归到深度网络