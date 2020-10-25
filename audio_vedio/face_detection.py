import cv2 as cv # opencv库
import dlib


# 人脸检测
path='cache/data/haarcascades/haarcascade_frontalface_default.xml'
detector = cv.CascadeClassifier(path)

# 读取图片
image = cv.imread('cache/1.jpg')
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
faces = detector.detectMultiScale(gray)
print(len(faces))
# 标记人脸
for x,y,w,h in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

cv.imshow('faces', image)
# 窗口暂停
cv.waitKey(0)

# 销毁窗口
cv.destroyAllWindows()