# 测试数据集,将数据还原成图片28*28 0代表黑色像素,1代表白色像素
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter  # 去除重复元素,统计元素出现的次数


# 制定还原图片的索引
def repicture(index: int):
    # 读取数据集
    pic_data = pd.read_csv("./KNN/digital_recognition/data/data.csv")

    if index < 0 or index > len(pic_data) - 1:
        print("索引超出范围")
        return

    # 获取指定索引的数据
    data = pic_data.iloc[:, 1:]
    label = pic_data.iloc[:, 0]

    # 找用户传入索引的图片标签,打印数据信息
    print("图片标签:", label.iloc[index])
    print("图片数据形状:", data.iloc[index].shape)
    print("Label分布:", Counter(label))

    # print("Debug: 图片数据:", data.iloc[index].values)
    pic = data.iloc[index].values.reshape(28, 28)  # 将数据还原成28*28的图片
    plt.imshow(pic, cmap="gray")
    plt.axis("off")  # 关闭坐标轴显示
    plt.savefig(f"./KNN/digital_recognition/repicture/{index}.png")  # 保存还原的图片
    plt.show()


if __name__ == "__main__":
    repicture(0)  # 还原第一张图片,csv第二行数据,索引为0,标签为1
