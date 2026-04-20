import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression

# 1. 加载数据
source_data = pd.read_csv("./LogisticRegression/customer_churn/data/churn.csv")
source_data.info()
print(source_data.head())

# 2. 数据预处理

# 处理字符串类型的特征
source_data = pd.get_dummies(source_data, columns=["gender", "Churn"])
source_data.info()
print(source_data.head())

# 剔除无用的特征
source_data.drop(columns=["gender_Male", "Churn_No"], inplace=True)
print("=" * 16)
source_data.info()
print(source_data.head())
# False： 不流失, True： 流失
print("数据分布:", source_data.Churn_Yes.value_counts())
print("特征名称:", source_data.columns)

# 可视化,柱状图,在每个 Contract_Month 分类内部，再按照 Churn_Yes 的取值继续拆分，并用不同颜色显示
sns.countplot(x="Contract_Month", data=source_data, hue="Churn_Yes")
plt.title("Contract_Month vs Churn_Yes")
plt.savefig("./LogisticRegression/customer_churn/Contract_Month_vs_Churn_Yes.png")
# plt.show()

# 3. 特征选择
target_column = "Churn_Yes"

# 所有候选特征
candidate_features = source_data.drop(columns=[target_column])
target = source_data[target_column]

# 计算每个特征与流失结果的相关系数，并按绝对值从高到低排序
feature_score_matrix = (
    source_data.corr(numeric_only=True)[[target_column]]
    .drop(index=target_column)
    .rename(columns={target_column: "correlation_with_churn"})
    .sort_values(
        by="correlation_with_churn", key=lambda column: column.abs(), ascending=False
    )
)
feature_score_matrix["abs_correlation"] = feature_score_matrix[
    "correlation_with_churn"
].abs()

print("特征与流失结果的影响矩阵:")
print(feature_score_matrix)

# 这里使用绝对相关系数作为筛选依据，阈值可按需要调整
correlation_threshold = 0.10
selected_features = feature_score_matrix[
    feature_score_matrix["abs_correlation"] >= correlation_threshold
].index.tolist()

X = candidate_features[selected_features]
y = target

print("建议用于后续训练的特征:", selected_features)
print("筛选后特征数量:", len(selected_features))
print("训练特征形状:", X.shape)
print("目标形状:", y.shape)

# 生成筛选后特征与目标的相关性矩阵，便于继续观察多特征关系
selected_feature_matrix = source_data[selected_features + [target_column]].corr(
    numeric_only=True
)
print("筛选后特征相关性矩阵:")
print(selected_feature_matrix)

# 保存筛选后的特征列表和相关性矩阵
feature_score_matrix.to_csv(
    "./LogisticRegression/customer_churn/feature_score_matrix.csv"
)
selected_feature_matrix.to_csv(
    "./LogisticRegression/customer_churn/selected_feature_matrix.csv"
)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)
print("训练集特征形状:", X_train.shape)
print("测试集特征形状:", X_test.shape)
print("训练集目标形状:", y_train.shape)
print("测试集目标形状:", y_test.shape)

# 5. 特征缩放(归一化,标准化等)

# 6. 模型训练
model = LogisticRegression(max_iter=1000, random_state=23)
model.fit(X_train, y_train)

# 7. 模型评估
print("评估测试集上的模型性能...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("模型评估结果:")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分数: {f1:.4f}")

print("\n分类报告:")
# macro avg: 宏平均,不考虑样本权重,直接求平均;适用于数据均衡的情况
# weighted avg: 加权平均,考虑样本权重,适用于数据不均衡的情况
print(classification_report(y_test, y_pred))
