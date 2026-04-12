import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. 读取数据集
pic_data = pd.read_csv("./KNN/digital_recognition/data/data.csv")

label = pic_data.iloc[:, 0]
data = pic_data.iloc[:, 1:]

print("标签分布:", Counter(label))
print("数据形状:", data.shape)

# 2. 数据预处理
# 检测是否有缺失值,存在缺失值删除
if data.isnull().values.any():
    print("数据集中存在缺失值,正在删除...")
    data = data.dropna()
    label = label[data.index]  # 同步删除标签中的对应行

print("数据预处理完成,当前标签分布:", Counter(label))
print("数据预处理完成,当前数据形状:", data.shape)
print("数据标签形状:", label.shape)

# 标准化(归一化)数据,将像素值0-255缩放到0-1之间
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
# 查看前5个样本中非0的归一化值
scaled_df = pd.DataFrame(data_scaled, columns=data.columns)
print("样本的非0归一化数据(每行最多显示20项):")
for idx in range(min(1, len(scaled_df))):
    non_zero = scaled_df.iloc[idx][scaled_df.iloc[idx] > 0]
    preview_items = non_zero.head(20)
    print(f"样本{idx} 非0像素数: {len(non_zero)}")
    print(preview_items.to_string())
    print("-" * 60)

# 3. 拆分数据集 startify=label参数确保训练集和测试集中标签分布相似
x_train, x_test, y_train, y_test = train_test_split(
    data_scaled, label, test_size=0.2, random_state=21, stratify=label
)

# 4. 特征工程(略)
# 箱线图
# 热力图相关矩阵

# 5. 模型训练
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# 6. 模型评估
y_pred = knn.predict(x_test)
"""

"""
print("分类报告:\n", classification_report(y_test, y_pred))

"""
混淆矩阵:confusion_matrix
行 = 真实标签，列 = 预测标签。
对角线数字是预测正确数，越大越好。
非对角线是错分到别的类.
"""
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("准确率:", accuracy_score(y_test, y_pred))

# 7. 模型持久化
model_dir = "./KNN/digital_recognition/model"
os.makedirs(model_dir, exist_ok=True)

# 同时保存模型、归一化器和特征顺序，确保推理阶段可复现训练预处理
bundle = {
    "model": knn,
    "scaler": scaler,
    "feature_columns": data.columns.tolist(),
}
model_path = os.path.join(model_dir, "digital_recognition.joblib")
joblib.dump(bundle, model_path)

# 简单加载验证
loaded_bundle = joblib.load(model_path)
print("模型已保存:", model_path)
print("保存内容:", list(loaded_bundle.keys()))
