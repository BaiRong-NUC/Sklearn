from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# KNN 有监督学习 -> 分类
print("=" * 60)
print("1. 加载数据集")
# 1. 加载数据集
iris = load_iris()

df_preview = pd.DataFrame(iris.data[:5], columns=iris.feature_names)
df_preview["target"] = iris.target[:5]
df_preview["target_name"] = [iris.target_names[t] for t in iris.target[:5]]

print("=" * 60)
print(f"{'鸢尾花数据集预览 (前5行)':^40}")
print("=" * 60)
print(df_preview.to_string(index=True))
print("-" * 60)
print(f"总数据量: {len(iris.data)} 条")
print("=" * 60)

print("2. 数据可视化")
# 2. 数据可视化
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target
iris_df["target_name"] = [iris.target_names[t] for t in iris.target]
print("\n数据集信息:")
print(iris_df.info())
print("\n数据集描述统计:")
print(iris_df.describe())
print("\n缺失值统计:")
print(iris_df.isnull().sum())

plt_x = "sepal length (cm)"
plt_y = "petal length (cm)"

# 散点图
sns.lmplot(
    x=plt_x,
    y=plt_y,
    data=iris_df,
    hue="target",
    palette="Set1",
    markers=["o", "s", "D"],
    fit_reg=False,  # 不画拟合线
)
plt.xlabel(plt_x)
plt.ylabel(plt_y)
plt.title("Iris Dataset - Sepal Length vs Petal Length")
plt.tight_layout()
# 保存图片
plt.savefig("./KNN/iris/iris_data_visualization.png", dpi=300)
# plt.show()

# 热力图
plt.figure(figsize=(8, 6))
sns.heatmap(
    iris_df[iris.feature_names + ["target"]].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
)
plt.title("Iris Dataset - Feature Correlation Heatmap")
plt.tight_layout()
# 保存图片
plt.savefig("./KNN/iris/iris_correlation_heatmap.png", dpi=300)
# plt.show()

# 箱线图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feature in zip(axes.flat, iris.feature_names):
    sns.boxplot(
        x="target_name",
        y=feature,
        data=iris_df,
        hue="target_name",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Species")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} Distribution by Species")

plt.suptitle("Iris Dataset - Boxplots for All Features", y=1.02)
plt.tight_layout()
# 保存图片
plt.savefig("./KNN/iris/iris_boxplot_all_features.png", dpi=300, bbox_inches="tight")
# plt.show()

print("=" * 60)
print("3. 数据集划分")
# 3. 数据集划分
# 3.1 数据预处理
null_counts = iris_df.isnull().sum()
if null_counts.sum() > 0:
    print("\n存在缺失值,正在使用平均数填充...")
    for col in iris_df.columns[:-2]:  # 只处理特征列
        if null_counts[col] > 0:
            mean_value = iris_df[col].mean()
            iris_df[col].fillna(mean_value, inplace=True)
    print("缺失值已填充。")


# 3.2 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

print(f"\n训练集样本数: {len(x_train)}")
print(f"测试集样本数: {len(x_test)}")

print("=" * 60)
print("4. 特征工程")
# 4. 特征工程
# 4.1 特征提取:原数据已经是数值型特征,无需提取

# 4.2 特征预处理(归一化/标准化)
scaler = StandardScaler()
x_train = scaler.fit_transform(
    x_train
)  # 先拟合训练数据并转换,适用于第一次使用该scaler对象,一般处理训练数据
x_test = scaler.transform(
    x_test
)  # 直接转换测试数据,适用于已经拟合过的scaler对象,一般处理测试数据

# 4.3 特征降维:原数据特征较少,无需降维

# 4.4 特征选择:原数据特征较少,无需选择

# 4.5 特征组合(不进行)

print(f"\n特征工程完成后,训练集:\n{x_train[:5]}\n测试集:\n{x_test[:5]}")
print("=" * 60)

print("5. 模型训练")
# 5. 模型训练与预测
# 5.1 模型训练
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# 5.2 模型评估
# 预测测试集
y_pred = knn.predict(x_test)

# 预测数据集外的新样本
x_new = scaler.transform([[7.8, 2.1, 3.9, 1.6]])  # 新样本特征
y_new_pred = knn.predict(x_new)  # 预测新样本类别
print(f"\n新样本特征: {x_new[0]},各类别概率: {knn.predict_proba(x_new)}")
print(f"新样本预测类别: {y_new_pred[0]} ({iris.target_names[y_new_pred[0]]})")
print("=" * 60)

# 6. 模型评估
print("6. 模型评估")
# print("训练正确率:", knn.score(x_test, y_test))
print("acc准确率:", accuracy_score(y_test, y_pred))

# 7. 超参调优
print("=" * 60)
print("7. 超参调优")

# 定义超参数网格
param_grid = {
    "n_neighbors": [1, 2, 3, 5, 7],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=4, scoring="accuracy")
grid_search.fit(x_train, y_train)

# 输出最佳超参数组合
print(f"最佳超参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最佳超参数组合训练模型
best_knn = grid_search.best_estimator_
best_knn.fit(x_train, y_train)

# 在测试集上评估模型性能
y_test_pred = best_knn.predict(x_test)
print(f"测试集准确率: {accuracy_score(y_test, y_test_pred):.4f}")

