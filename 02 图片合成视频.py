import cv2

# 1. 读取一张图片，获取info  2. 创建视频读取对象  3. 写入视频对象中

# 读取图片
img = cv2.imread("./image1.jpg")
imageInfo = img.shape
size = (imageInfo[0], imageInfo[1])

# 写入对象的创建： 参数： 1.文件名  2.编码器   3. 帧率（每秒显示图片数） 4.大小
videoWrite = cv2.VideoWriter("image_2_video.mp4", -1, 5, size)

for i in range(1, 11):
    img = cv2.imread("./image{num}.jpg".format(num = i))
    videoWrite.write(img)  # 写入方法  1  jpg  data
print("END!")

# 备注：
#  1. 不同于其他文件读写，视频写入对象先指定保存的文件名
#  2.