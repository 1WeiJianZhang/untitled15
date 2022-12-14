'''
深度学习代码中的随机种子
深度学习网络模型中初始的权值参数通常都是初始化成随机数
而使用梯度下降法最终得到的局部最优解对于初始位置点的选择很敏感
为了能够完全复现作者的开源深度学习代码，随机种子的选择能够减少一定程度上
算法结果的随机性，也就是更接近于原始作者的结果
即产生随机种子意味着每次运行实验，产生的随机数都是相同的
但是在大多数情况下，即使设定了随机种子，仍然没有办法完全复现
作者paper中所给出的模型性能，这是因为深度学习代码中除了产生随机数中带有随机性
其训练的过程中使用 mini-batch SGD或者优化算法进行训练时，本身就带有了随机性
因为每次更新都是从训练数据集中随机采样出batch size个训练样本计算的平均梯度
作为当前step对于网络权值的更新值，所以即使提供了原始代码和随机种子，想要
复现作者paper中的性能也是非常困难的
'''
# pytorch中的随机种子
import torch

torch.manual_seed(5)
a = torch.randn(2, 3)

print(a)
'''
每次运行实验，tensor a都是这个结果
tensor([[-0.4868, -0.6038, -0.5581],
        [ 0.6675, -0.1974,  1.9428]])
'''

# numpy中的随机种子
import numpy as np

np.random.seed(15)
b = np.random.rand(5)
print(b)
# [0.8488177  0.17889592 0.05436321 0.36153845 0.27540093]

# random 模块中的随机种子，random是python中用于产生随机数的模块
import random

random.seed(10)
print(random.random())
# 0.5714025946899135