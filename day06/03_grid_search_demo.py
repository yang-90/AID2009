# 03_grid_search_demo.py
# 网格搜索示例
""" 网格搜索：
利用穷举法，将不同参数取值进行组合，挑选出最优
的参数组合，简化模型参数确定
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

x, y = [], []
with open("../data/multiple2.txt", "r") as f:
    for line in f.readlines():
        data = [float(substr)
                for substr in line.split(",")]
        x.append(data[:-1])  # 输入
        y.append(data[-1])  # 输出

x = np.array(x)
y = np.array(y, dtype=int)

# 定义可选参数列表
params = [
    {
        "kernel":["linear"],
        "C":[1, 10, 100, 1000]
    },
    {
        "kernel":["poly"],
        "C":[1],
        "degree":[2, 3]
    },
    {
        "kernel":["rbf"],
        "C":[1, 10, 100, 1000],
        "gamma":[1, 0.1, 0.01, 0.001]
    }
]

model = ms.GridSearchCV(svm.SVC(),
                        params)
model.fit(x, y) # 训练
# 打印最优参数、最好成绩
print("best_score_:", model.best_score_) # 最好成绩
print("best_params_:", model.best_params_) # 最优组合







