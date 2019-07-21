"""
    1. load xml
    2. load image
    3. gray   compute haar特征(交给openCV)
    4. detect
    5. draw
"""

import cv2
import numpy as np

def main():

    # 加载人脸识别分类器
    face_xml = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # 加载eye识别分类器
    eye_xml = cv2.CascadeClassifier("haarcascade_eye.xml")

    # 加载图片
    img = cv2.imread("face.jpg")
    cv2.imshow("src", img)
    # gray   (haar略过)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect   参数： 1.待检测灰度图  2.缩放比例（haar模板）  3. 人脸不低于5个像素值
    faces = face_xml.detectMultiScale(gray, 1.3, 5)
    print("faces = ", len(faces))

    # draw
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        # 人脸区域作为roi
        roi_face = gray[y:y+h, x:x+w]   # 灰度图
        roi_color = img[y:y+h, x:x+w]   # 彩色图
        # 参数：灰度图
        eyes = eye_xml.detectMultiScale(roi_face)
        print("eyes=", eyes)
        for ex, ey, ew, eh in eyes:
            # 在彩色图上进行绘制
            # 为什么不是在img上绘制，而在roi区域中绘制？
            # 答：在eye中识别是roi区域，并不是原图中的位置
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    cv2.imshow("dst", img)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
