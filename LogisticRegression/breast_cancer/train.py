import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 1. 加载数据
source_data = pd.read_csv("./LogisticRegression/breast_cancer/data/data.csv")
print(source_data.head())

data = source_data.iloc[:, :-1]
target = source_data.iloc[:, -1]
print("数据形状:", data.shape)
print("目标形状:", target.shape)

# 2. 数据预处理

# 检测数据是否有缺失值
for column in data.columns:
    missing_count = data[column].isnull().sum()
    print(f"{column} 列缺失值数量: {missing_count}")

# 检测并丢弃异常字符所在行（把特征列转换为数值，无法转换的视为异常）
numeric_data = data.apply(pd.to_numeric, errors="coerce")
invalid_mask = numeric_data.isna() & data.notna()
invalid_rows_mask = invalid_mask.any(axis=1)

invalid_count = invalid_rows_mask.sum()
print(f"检测到包含异常字符的行数: {invalid_count}")

if invalid_count > 0:
    print("以下为包含异常字符的样本（前 10 行）:")
    print(source_data.loc[invalid_rows_mask].head(10))

    # 同步删除异常行
    data = numeric_data.loc[~invalid_rows_mask].copy()
    target = target.loc[~invalid_rows_mask].copy()
else:
    data = numeric_data.copy()

print("清洗后数据形状:", data.shape)
print("清洗后目标形状:", target.shape)

# 如果仍有缺失值，这里直接删除对应样本
clean_mask = data.notna().all(axis=1)
if (~clean_mask).sum() > 0:
    data = data.loc[clean_mask].copy()
    target = target.loc[clean_mask].copy()
    print("删除缺失值后数据形状:", data.shape)

data.info()

# 4. 特征提取
X = data.iloc[:, 1:]  # 第一列是ID，没有实际意义
y = target
print("特征形状:", X.shape)
print("目标形状:", y.shape)

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征归一化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 5. 模型训练
model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(x_train, y_train)

# 6. 模型评估
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)

# 本数据集标签为 2(良性) 和 4(恶性)，显式指定 4 为正类
positive_label = 4

# 精确率: TP / (TP + FP)
precision = precision_score(y_test, y_pred, pos_label=positive_label)
print("模型精确率:", precision)

# 召回率: TP / (TP + FN)
recall = recall_score(y_test, y_pred, pos_label=positive_label)
print("模型召回率:", recall)

# f1-score: 2 * (精确率 * 召回率) / (精确率 + 召回率)
f1 = f1_score(y_test, y_pred, pos_label=positive_label)
print("模型F1分数:", f1)
