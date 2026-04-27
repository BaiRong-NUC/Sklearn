import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 加载数据
src_data = pd.read_csv("./DecisionTree/Survive/data/data.csv")
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
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# 预测测试集结果
y_pred = model.predict(x_test)

# 评估模型性能
print(f"模型评估结果:\n{classification_report(y_test, y_pred)}")

# 输出混淆矩阵
print(f"混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")

# 输出决策树(默认dip=100)
plt.figure(figsize=(30, 20), dpi=100)
plot_tree(
    model,
    filled=True,  # 节点颜色填充
    feature_names=X.columns,  # 特征名称
    class_names=["Not Survived", "Survived"],  # 类别名称
    max_depth=10,  # 限制树的最大深度以便可视化
)
plt.title("decision_tree")
plt.savefig("./DecisionTree/Survive/decision_tree.png")  # 保存决策树图像
plt.show()
