# 02_validation_curve_demo.py
# 验证曲线示例
# 验证曲线：验证不同参数下模型的性能
#          如决策树的深度、随机森林树的数量等等
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms
import matplotlib.pyplot as mp

data = []
with open("../data/car.txt", "r") as f:
    for line in f.readlines():
        data.append(line.replace("\n", "").split(","))

data = np.array(data).T  # 转置
encoders, train_x = [], []

# 对样本数据进行标签编码
for row in range(len(data)):
    encoder = sp.LabelEncoder()  # 创建标签编码器
    encoders.append(encoder)
    if row < len(data) - 1:  # 不是最后一行，为样本特征
        lbl_code = encoder.fit_transform(data[row])  # 编码
        train_x.append(lbl_code)
    else:  # 最后一行，为样本输出
        train_y = encoder.fit_transform(data[row])

train_x = np.array(train_x).T  # 转置回来，变为编码后的矩阵

# 定义模型
# n_estimators参数是要验证的参数，故不设置
model = se.RandomForestClassifier(
    max_depth=8, # 最大深度
    random_state=7) # 随机种子
# 产生待验证参数数组
n_estimators = np.arange(50, 550, 50)
print(n_estimators)

# 利用验证曲线，验证不同树数量的随机森林
train_scores1, test_scores1 = ms.validation_curve(
    model, # 待验证的模型
    train_x, train_y, # 数据集
    "n_estimators", # 待验证的参数名称
    n_estimators, # 待验证参数的可选值
    cv=5) # 折叠数量
train_mean = train_scores1.mean(axis=1)
test_mean = test_scores1.mean(axis=1)
print("train_mean:", train_mean)
print("test_mean:", test_mean)

# 可视化
mp.figure('n_estimators', facecolor='lightgray')
mp.title('n_estimators', fontsize=20)
mp.xlabel('n_estimators', fontsize=14)
mp.ylabel('F1 Score', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(n_estimators, test_mean, 'o-',
        c='blue', label='Testing')
mp.legend()
mp.show()