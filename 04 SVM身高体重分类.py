"""
    身高体重 预测  男女

    SVM：
        1. SVM_create（）
        2. svm.train()
        3. svm.predict()

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():

    # 1. 准备数据
    rand_girl = np.array([[155, 48], [159, 50], [164, 53], [168, 56], [172, 60]])
    rand_boy = np.array([[152, 53], [156, 55], [160, 56], [172, 64], [176, 65]])

    # 2. label
    label = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

    # 3 data
    data = np.vstack((rand_girl, rand_boy))
    data = np.array(data, dtype="float32")

    # SVM 所有数据都需要有标签（监督学习）
    # [155, 48] -- 0   女生    [152, 53]  -- 1  男生

    # 4. 训练
    svm = cv2.ml.SVM_create()   # ml: machine learning   svm_create():创建支持向量机
    # 属性设置
    svm.setType(cv2.ml.SVM_C_SVC)   # SVM类型
    svm.setKernel(cv2.ml.SVM_LINEAR)  # 线性分类器
    svm.setC(0.01)  # 核相关参数
    result = svm.train(data, cv2.ml.ROW_SAMPLE, label)
    print(result)   # bool  True:训练成功  False:训练失败

    # 预测（验证预测效果）
    pt_data = np.vstack(([167, 55], [162, 57]))    # 矩阵1：女生（0） 矩阵2：男生（2）
    pt_data = np.array(pt_data, dtype="float32")
    par1, par2 = svm.predict(pt_data)
    print(par1, "\n", par2)
    # par1: 0(表示什么意思？)
    # par2: 预测结果


if __name__ == "__main__":
    main()