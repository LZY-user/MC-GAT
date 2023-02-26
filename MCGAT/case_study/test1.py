# data1 = open("case1.label")
# data = open("train1.txt")
# num1 = []
#
# for line in data1.readlines():
#     line = int(line)
#     num1.append(line)
#
# for line in data.readlines():
#     line = int(line)
#     print(num1[line])


# import torch.nn.functional as F
# import torch
#
#
# # 手动实现 NLLLoss() 函数功能
# data = torch.randn(5, 5)  # 随机生成一组数据
# target = torch.tensor([0, 2, 4, 3, 1])  # 标签
# one_hot = F.one_hot(target).float()  # 对标签作 one_hot 编码
#
# exp = torch.exp(data)  # 以e为底作指数变换
# sum = torch.sum(exp, dim=1).reshape(-1, 1)  # 按行求和
# softmax = exp / sum  # 计算 softmax()
# log_softmax = torch.log(softmax)  # 计算 log_softmax()
# nllloss = -torch.sum(one_hot * log_softmax) / target.shape[0]  # 标签乘以激活后的数据，求平均值，取反
# print("nllloss:", nllloss)
#
#
# # 调用 NLLLoss() 函数计算
# Log_Softmax = F.log_softmax(data, dim=1)  # log_softmax() 激活
# Nllloss = F.nll_loss(Log_Softmax, target)  # 无需对标签作 one_hot 编码
# print("Nllloss:", Nllloss)
#
#
# # 直接使用交叉熵损失函数 CrossEntropy_Loss()
# cross_entropy = F.cross_entropy(data, target)  # 无需对标签作 one_hot 编码
# print('cross_entropy:', cross_entropy)

import scipy.sparse as sp
import numpy as np


a = np.array([[0,3,2],[0,4,5],[1,3,6],[1,5,7],[2,4,10]])
print(a)
b = sp.coo_matrix(arg1=(a[:, 2], (a[:, 0], a[:, 1])), shape=(7,7), dtype=np.float32)
print(b)
c = b.todense()
print(c)
