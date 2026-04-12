import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def preprocess_image(image_path: str) -> np.ndarray:
    img = plt.imread(image_path)

    # 训练特征是 28*28=784 维
    if img.shape != (28, 28):
        raise ValueError(f"图片尺寸应为 28x28,当前是 {img.shape}")

    return img.reshape(1, -1)  # 转换为 (1, 784) 的二维数组，适配模型输入


def main():
    model_path = "./KNN/digital_recognition/model/digital_recognition.joblib"
    image_path = "./KNN/digital_recognition/data/my_input.png"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}，请先运行 train.py")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到输入图片: {image_path}")

    bundle = joblib.load(model_path)
    knn = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle.get("feature_columns")

    x = preprocess_image(image_path)
    if feature_columns:
        x_input = pd.DataFrame(x, columns=feature_columns)
    else:
        x_input = x

    x_scaled = scaler.transform(x_input)
    # x_scaled = x_input  # 图片读取时已经是0-1的像素值,不需要再缩放了,否则会导致预测结果不正确
    pred = knn.predict(x_scaled)[0]

    print(f"输入图片: {image_path}")
    print(f"预测结果: {pred}")


if __name__ == "__main__":
    main()
