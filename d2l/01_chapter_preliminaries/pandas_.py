##########################################################################################################################
# 为了能用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始， 而不是从那些准备好的张量格式数据开始。 在Python中常用的数据分析工具中，我们
# 通常使用pandas软件包。 像庞大的Python生态系统中的许多其他扩展包一样，pandas可以与张量兼容。 本节我们将简要介绍使用pandas预处理原始数据，并将
# 原始数据转换为张量格式的步骤。 后面的章节将介绍更多的数据预处理技术。
# ##########################################################################################################################

# 1. 创建一个人工数据集，并存储在CSV（逗号分隔值）文件
import os
# !pip install pandas
import pandas as pd
import torch

os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 从创建的CSV文件中加载原始数据集]
data = pd.read_csv(data_file)
print(data)


# 处理缺失值:“NaN”项代表缺失值。 [为了处理缺失的数据，典型的方法包括插值法和删除法，] 
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean())
# print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转换为张量格式
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)