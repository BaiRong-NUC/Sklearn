import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()
