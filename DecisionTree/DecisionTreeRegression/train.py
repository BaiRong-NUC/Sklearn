import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x_train = np.array(list(range(1, 11))).reshape(-1, 1)
y_train = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.5])

# 决策树回归模型训练
model = DecisionTreeRegressor(random_state=42)
model_3 = DecisionTreeRegressor(
    random_state=42, max_depth=3
)  # 限制树的最大深度为3，防止过拟合
# 线性回归模型训练
line_model = LinearRegression()

# 模型训练
model.fit(x_train, y_train)
model_3.fit(x_train, y_train)
line_model.fit(x_train, y_train)

x_test = np.arange(0, 10, 0.1).reshape(-1, 1)
print(f"测试集特征:\n{x_test[:5]} ...\n{x_test[-5:]}")
y_pred = model.predict(x_test)
y_pred_3 = model_3.predict(x_test)
line_pred = line_model.predict(x_test)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color="red", label="Training Data")
plt.plot(x_test, y_pred, color="blue", label="Decision Tree Prediction")
plt.plot(x_test, line_pred, color="green", label="Linear Regression Prediction")
plt.plot(
    x_test, y_pred_3, color="orange", label="Decision Tree (max_depth=3) Prediction"
)
plt.title("Decision Tree Regression vs Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
