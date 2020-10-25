# -- coding: utf-8 --
from moviepy.editor import *
import tensorflow as tf
import cv2 as cv
from scipy.io import wavfile
import dlib

# video = VideoFileClip('cache/test.avi')
# audio = video.audio
# audio.write_audiofile('cache/test.wav')

#detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1('cache/mmod_human_face_detector.dat/mmod_human_face_detector.dat')
predector_landmarks=dlib.shape_predictor('cache/face_landmarks.dat')

capture=cv.VideoCapture('cache/test.avi')
while True:
    ret,frame=capture.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces=detector(gray,1)
        for face in faces:
            left = face.rect.left()
            top = face.rect.top()
            right = face.rect.right()
            bottom = face.rect.bottom()
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        cv.imshow('audio',frame)
        cv.waitKey(10)

    else:
        break
