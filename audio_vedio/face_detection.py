import cv2  # opencv库
import dlib


# 人脸检测
face_detect = dlib.get_frontal_face_detector()
predector_landmarks=dlib.shape_predictor('cache/face_landmarks.dat')

# 读取图片
image = cv2.imread('cache/1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faces = face_detect(gray,1)
print(len(faces))
# 标记人脸
for i,face in enumerate(faces):
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

cv2.imshow('faces', image)
# 窗口暂停
cv2.waitKey(0)

# 销毁窗口
cv2.destroyAllWindows()