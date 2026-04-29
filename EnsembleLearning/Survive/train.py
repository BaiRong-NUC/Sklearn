import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 加载数据
src_data = pd.read_csv("./EnsembleLearning/Survive/data/data.csv")
src_data.info()

# 缺失值处理
src_data["Age"] = src_data["Age"].fillna(src_data["Age"].mean())

# Embarked 缺失值填充 + 数值编码标注
embarked_map = {"C": 0, "Q": 1, "S": 2}
src_data["Embarked"] = (
    src_data["Embarked"]
    .fillna(src_data["Embarked"].mode()[0])
    .map(embarked_map)
    .astype(int)
)
print("\n\n缺失值处理后数据概况:\n\n")
src_data.info()

# 字符串热编码
src_data = pd.get_dummies(src_data, columns=["Sex"])
src_data = src_data.drop(columns=["Name", "Ticket", "Cabin", "Sex_male"])
src_data = src_data.rename(columns={"Sex_female": "Sex"})

# 导出带编码标注的数据集（Embarked: C=0, Q=1, S=2）
# src_data.to_csv("./DecisionTree/Survive/data/data_encoded.csv", index=False)
print("\nEmbarked 编码标注: C=0, Q=1, S=2")
# print("已导出编码后数据集: ./DecisionTree/Survive/data/data_encoded.csv")
print("\n\n字符串热编码后数据概况:\n\n")
src_data.info()

# 划分特征和标签
X = src_data.drop(columns=["Survived"])
y = src_data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练决策树模型
print("单一决策树模型")
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(x_train, y_train)

# 预测测试集结果
y_pred = model_tree.predict(x_test)

# 评估模型性能
print(f"模型评估结果:\n{classification_report(y_test, y_pred)}")

# 输出混淆矩阵
print(f"混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")

# 输出准确率
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

# 随机森林模型训练(默认参数)
print("\n\n随机森林(默认参数)模型")
model_rf_ = RandomForestClassifier(random_state=42)
model_rf_.fit(x_train, y_train)
y_pred = model_rf_.predict(x_test)
print(f"随机森林模型评估结果:\n{classification_report(y_test, y_pred)}")
print(f"随机森林混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
print(f"随机森林准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n\n随机森林(超参数调优)模型")

# 网格搜索超参数调优
param_grid = {
    "n_estimators": [60, 90, 100, 200, 300],  # 森林中树的数量
    "max_depth": [None, 3, 5, 7, 10, 20],  # 树的最大深度
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5折交叉验证
    n_jobs=-1,  # 使用所有CPU核心
    scoring="f1",
    # verbose=2,  # 输出详细日志
)

grid_search.fit(x_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)
print(f"调优后随机森林模型评估结果:\n{classification_report(y_test, y_pred)}")
print(f"调优后随机森林混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
print(f"调优后随机森林准确率: {accuracy_score(y_test, y_pred):.4f}")
