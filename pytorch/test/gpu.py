# 对比cpu和gpu的性能
import torch
import time

print(torch.__version__)
print(torch.cuda.is_available())

a = torch.randn(10000, 1000)
# print(a)
b = torch.randn(1000, 2000)
# print(b)

t0 = time.time()
# Matrix product of two tensors.
c = torch.matmul(a, b)
t1 = time.time()

print(a.device, t1-t0, c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

# 包含了初始化的时间
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()

print(a.device, t2-t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()

print(a.device, t2-t0, c.norm(2))

# 2.4.1+cu121
# True
# cpu 0.19435429573059082 tensor(141506.2500)
# cuda:0 0.01994180679321289 tensor(141650.3125, device='cuda:0')
# cuda:0 0.00021314620971679688 tensor(141650.3125, device='cuda:0')