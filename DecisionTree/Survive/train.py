import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 加载数据
src_data = pd.read_csv("./DecisionTree/Survive/data/data.csv")
src_data.info()
