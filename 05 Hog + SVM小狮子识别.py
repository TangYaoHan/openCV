"""
    1. 样本（正负  ->  1:2或1:3）
            pos: 正样本，包含obj， 尽可能多样性（在干扰多样条件训练，识别效果更好）
            neg：负样本，一定不能包含obj（尽量干扰较多，如更换背景，相似目标）
            size： 64*128
            如何获取样本：   1. 网络    2. 公司内部   3. 自己收集（网络爬虫）
            图像处理： 裁剪、缩放（保持同样的尺寸）
    2. 训练
    3. 预测
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 1.参数设置  2.hog  3.svm  4. computer  hog  5.label   6.训练  7.预测

# 1.参数设置
PosNum = 820
NegNum = 1931
winSize = (64, 128)
blockSize = (16, 16)   # 105
blockStride = (8, 8)
cellSize = (8, 8)  # 4 cell
nbin = 9  # 9 bin

# 2. hog创建
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbin)

# 3. svm分类器创建
svm = cv2.ml.SVM_create()

# 4. computer hog
featureNum = int(((128-16)/8+1)*((64-16)/8+1))*4*9  # 3780
featureArray = np.zeros(((PosNum + NegNum), featureNum), np.float32)
labelArray = np.zeros(((PosNum+NegNum), 1), np.int32)

# 5. svm 监督学习  样本  标签  ->  image hog
for i in range(0, PosNum):
    fileName = "./pos/" + str(i+1) + ".jpg"
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8, 8))   # 3780维
    for j in range(0, featureNum):
        featureArray[i, j] = hist[j]
        labelArray[i, 0] = 1  # 正样本
for i in range(0, NegNum):
    fileName = "./neg/" + str(i + 1) + ".jpg"
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8, 8))  # 3780维
    for j in range(0, featureNum):
        featureArray[i+PosNum, j] = hist[j]
        labelArray[i+PosNum, 0] = -1  # 负样本

# svm属性设置

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

# 6. 训练
ret = svm.train(featureArray, cv2.ml.ROW_SAMPLE, labelArray)

# 7. 检测   核心：create myHog -> myDetect  -> array ->> resultArray
#           ->   resultArray = -1*alphaArray*supportArray   rho -> svm  -> svm.train
alpha = np.zeros([1], np.float32)
rho = svm.getDecisionFunction(0, alpha)
print(rho)
print(alpha)
alphaArray = np.zeros((1, 1), np.float32)
supportArray = np.zeros((1, featureNum), np.float32)
resultArray = np.zeros((1, featureNum), np.float32)
alphaArray[0, 0] = alpha
resultArray = -1*alphaArray*supportArray
# detect
myDetect = np.zeros((3781), np.float32)
for i in range(0, 3780):
    myDetect[i] = resultArray[0, i]
myDetect[3780] = rho[0]   # rho用于svm判决
# 构建hog
myHog = cv2.HOGDescriptor()
myHog.setSVMDetector(myDetect)

# 读取待检测图片
# image_obj = cv2.imread("./Test2.jpg")
image_obj = cv2.imread("./face.jpg")
# 参数：（8,8）：window滑动步长  (32, 32):窗体大小    1.05:缩放系数
objs = myHog.detectMultiScale(image_obj, 0, (8, 8), (32, 32), 1.05, 2)
x = int(objs[0][0][0])
y = int(objs[0][0][1])
w = int(objs[0][0][2])
h = int(objs[0][0][3])
cv2.rectangle(image_obj, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("dst", image_obj)
cv2.waitKey(0)




