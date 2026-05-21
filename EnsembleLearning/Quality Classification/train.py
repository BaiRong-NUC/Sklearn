import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# 1. 加载数据
data = pd.read_csv("./EnsembleLearning/Quality Classification/data/data.csv")
data.info()
print("类别分布:\n", data["Class label"].value_counts())

# 修改标签为0-2
data["Class label"] = data["Class label"].map({1: 0, 2: 1, 3: 2})
print("修改后的类别分布:\n", data["Class label"].value_counts())

# 训练单一决策树AdaBoost模型(只适用于二分类问题)
data = data[data["Class label"] != 2]
print("删除 class label=2 后的类别分布:\n", data["Class label"].value_counts())

# 2. 划分训练集和测试集
x_data = data.drop("Class label", axis=1)

y = data["Class label"]

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 训练单一决策树AdaBoost模型(只适用于二分类问题)

adaBoostClass = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
)

adaBoostClass.fit(x_train, y_train)
y_pred = adaBoostClass.predict(x_test)
# print("单一决策树AdaBoost模型预测结果:\n", y_pred)
print("单一决策树AdaBoost模型评估结果:\n", classification_report(y_test, y_pred))
