# 01_cross_validation_demo.py
# 交叉验证示例
# 交叉验证：将数据集划分成k个折叠(子集)，轮流使用
#         其中一个折叠作为测试集，其它作为训练集
#         这样相当于获得了k个测试集、训练集
#         主要用于样本数量较少的情况
# 交叉验证示例
import numpy as np
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp

x, y = [], []  # 输入，输出

# 读取数据文件
with open("../data/multiple1.txt", "r") as f:
    for line in f.readlines():
        data = [float(substr) for
                substr in line.split(",")]
        x.append(data[:-1])  # 输入样本：取从第一列到导数第二列
        y.append(data[-1])  # 输出样本：取最后一列

x = np.array(x)
y = np.array(y, dtype=int)

# 定义模型
model = nb.GaussianNB()

# 做交叉验证，验证每个折叠下的指标
precision = ms.cross_val_score(
    model,  # 模型
    x, y,  # 原始样本
    cv=5,  # 划分为5个折叠
    scoring="precision_weighted")  # 查准率
print("查准率:", precision.mean())


recall = ms.cross_val_score(
    model,  # 模型
    x, y,  # 原始样本
    cv=5,  # 划分为5个折叠
    scoring="recall_weighted")  # 召回率
print("召回率:", recall.mean())

f1 = ms.cross_val_score(
    model,  # 模型
    x, y,  # 原始样本
    cv=5,  # 划分为5个折叠
    scoring="f1_weighted")  # F1
print("F1:", f1.mean())


