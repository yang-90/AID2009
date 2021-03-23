# 04_kmeans_demo.py
# k-means聚类示例
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x = [] # 样本输入(无监督学习没有输出)
with open("../data/multiple3.txt", "r") as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        data = [float(substr)
                for substr in line.split(",")]
        x.append(data)
x = np.array(x) # 列表转数组

# 定义模型
model = sc.KMeans(n_clusters=4) # 聚类数量为4
model.fit(x) # 训练

# 取出聚类结果和中心
pred_y = model.labels_ # 聚类结果
centers = model.cluster_centers_ # 聚类中心
print("聚类结果:", pred_y)
print("聚类中心:", centers)

# 绘制聚类结果
mp.figure("k-means")
mp.title("k-means")
mp.xlabel("x", fontsize=14)
mp.ylabel("y", fontsize=14)
# 原始样本散点图
mp.scatter(x[:, 0], x[:, 1], # 散点的x,y坐标数据
           c=pred_y, # 每个样本的聚类
           cmap="brg") # 不同样本涂成不同色彩
# 绘制聚类中心
mp.scatter(centers[:, 0], centers[:, 1],#中心x,y坐标数据
           marker="+", # 形状为+
           c="black", s=200, linewidths=1)#颜色、大小、线条
mp.show()