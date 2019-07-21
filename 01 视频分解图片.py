# 视频分解图片
# 1. load  2. infor  3. parse  4. imshow imwrite

import cv2

cap = cv2.VideoCapture("lion_video.mp4")
isOpened = cap.isOpened
print(isOpened)

fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0  # 给图片命名
while isOpened:
    if i == 10:
	    break
    else:
	    i = i + 1
    flag, frame = cap.read()  # 读取每一张图片  flag:读取成功或失败， frame:图片信息
    fileName = "image" + str(i) + ".jpg"
    if flag:
        cv2.imwrite(fileName, frame)
        # JEPG: 100质量最高    PNG：0-9等级，等级越高，质量越差

print("END!")