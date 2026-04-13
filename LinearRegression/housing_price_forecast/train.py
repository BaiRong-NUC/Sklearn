import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
)

# 1. 加载数据
source_data = pd.read_csv("./LinearRegression/housing_price_forecast/data/data.csv")

print(source_data.head())

data = source_data.iloc[:, :-1]
price = source_data.iloc[:, -1]
print("数据形状:", data.shape)
print("价格形状:", price.shape)

# 2. 丢弃缺失值
data = data.dropna()
price = price.dropna()
print("丢弃缺失值后数据形状:", data.shape)
print("丢弃缺失值后价格形状:", price.shape)

# 3. 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(
    data, price, test_size=0.2, random_state=23
)

# 3. 特征工程
transfor = StandardScaler()
x_train = transfor.fit_transform(x_train)
x_test = transfor.transform(x_test)

# 4. 模型训练

# 正规方程法:
# line_model = LinearRegression(fit_intercept=True)
# 梯度下降法:
line_model = SGDRegressor(
    loss="squared_error", fit_intercept=True, learning_rate="invscaling", eta0=0.01
)

line_model.fit(x_train, y_train)
print("权重:", line_model.coef_)
print("偏置:", line_model.intercept_)

# 5. 模型评估
y_pred = line_model.predict(x_test)
print("均方误差:", mean_squared_error(y_test, y_pred))
print("均方根误差:", root_mean_squared_error(y_test, y_pred))
print("平均绝对误差:", mean_absolute_error(y_test, y_pred))
print(
    "R² 决定系数:", r2_score(y_test, y_pred)
)  #  衡量模型对数据的拟合程度 1表示完美拟合，0表示没有拟合
