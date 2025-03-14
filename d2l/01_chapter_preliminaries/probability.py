##########################################################################################################################
# 基本概率论
# ##########################################################################################################################
import torch
# https://devdocs.io/pytorch~2-probability-distributions/
from torch.distributions import multinomial
from d2l import torch as d2l


'''
https://devdocs.io/pytorch~2/distributions#torch.distributions.multinomial.Multinomial
在 PyTorch 中，torch.distributions.multinomial.Multinomial 是一个用于创建多项分布对象的类，多项分布是二项分布在多个类别上的推广，常用于模拟在多次独立试验中，每个类别出现的次数。
torch.distributions.multinomial.Multinomial(total_count=1, probs=None, logits=None, validate_args=None)
参数说明：
total_count：试验的总次数，必须为非负整数，默认为 1。
probs：每个类别出现的概率，是一个非负张量，且最后一维的元素之和必须为 1。
logits：每个类别出现的对数概率，与 probs 二选一。如果提供了 logits，则通过 softmax 函数将其转换为概率。
validate_args：是否进行参数验证，默认为 None。
'''
# 定义每个类别的概率
probs = torch.tensor([0.2, 0.3, 0.5])
print("probs = ", probs)

# 创建多项分布对象
m = multinomial.Multinomial(total_count=10, probs=probs)
print("m = ", m)

# 采样
samples = m.sample()
print("samples = ", samples)

# 定义一个样本
value = torch.tensor([2, 3, 5])
# 计算对数概率
log_prob = m.log_prob(value)
print("log_prob = ", log_prob)

# 为了抽取一个样本，即掷骰子，我们只需传入一个概率向量。 输出是另一个相同长度的向量：它在索引处的值是采样结果中出现的次数
fair_probs = torch.ones([6]) / 6

print(fair_probs)

print(multinomial.Multinomial(1, fair_probs).sample())

multinomial.Multinomial(10, fair_probs).sample()

# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()