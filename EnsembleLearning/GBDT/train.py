import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# 1. 加载数据
data = pd.read_csv("./EnsembleLearning/GBDT/data/data.csv")
data.info()

# 缺失值处理
data["Age"] = data["Age"].fillna(data["Age"].mean())

# Embarked 缺失值填充 + 数值编码标注
embarked_map = {"C": 0, "Q": 1, "S": 2}
data["Embarked"] = (
    data["Embarked"].fillna(data["Embarked"].mode()[0]).map(embarked_map).astype(int)
)
# print("\n\n缺失值处理后数据概况:\n\n")
# data.info()


# 字符串热编码
src_data = pd.get_dummies(data, columns=["Sex"])
src_data = src_data.drop(columns=["Name", "Ticket", "Cabin", "Sex_male"])
src_data = src_data.rename(columns={"Sex_female": "Sex"})
src_data.info()

# 划分特征和标签
X = src_data.drop(columns=["Survived"])
y = src_data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 训练GBDT模型
print("GBDT模型")
model_gbdt = GradientBoostingClassifier(random_state=42)
model_gbdt.fit(x_train, y_train)

# 预测测试集结果
y_pred = model_gbdt.predict(x_test)
print(classification_report(y_test, y_pred))


print("-" * 30)
print("超参数调优后的GBDT模型")
pram_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
}
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=pram_grid,
    cv=5,
    n_jobs=-1,
    verbose=0,  # 不显示训练过程中的日志信息
)
grid_search.fit(x_train, y_train)
print(f"最佳超参数组合: {grid_search.best_params_}")
best_gbdt = grid_search.best_estimator_
y_pred_best = best_gbdt.predict(x_test)
print(classification_report(y_test, y_pred_best))
