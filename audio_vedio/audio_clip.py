# -- coding: utf-8 --
from moviepy.editor import *
import tensorflow as tf
import cv2 as cv
from scipy.io import wavfile
import dlib,time

# video = VideoFileClip('cache/test.avi')
# audio = video.audio
# audio.write_audiofile('cache/test.wav')

#detector = dlib.get_frontal_face_detector()
def detectface():
    detector = dlib.cnn_face_detection_model_v1('cache/mmod_human_face_detector.dat/mmod_human_face_detector.dat')
    #predector_landmarks=dlib.shape_predictor('cache/face_landmarks.dat')

    capture=cv.VideoCapture('cache/test.avi')
    fps=capture.get(cv.CAP_PROP_FPS)
    waittime=int(60.0/fps)
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
            key = cv.waitKey(waittime)
            if key == 27:
                break

        else:
            break


def detectface2():
    #path = 'cache/data/haarcascades_cuda/haarcascade_frontalface_default.xml'
    path = 'cache/data/haarcascades_cuda/haarcascade_frontalface_default.xml'
    detector = cv.CascadeClassifier(path)

    capture = cv.VideoCapture('cache/test.avi')
    fps=capture.get(cv.CAP_PROP_FPS)
    waittime=1/fps
    print(waittime)
    while True:
        start = time.time()
        ret, frame = capture.read()
        key=cv.waitKey(1)
        if key==27:
            break
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(gray)
            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.imshow('audio', frame)
            t=waittime-(time.time()-start)
            if t>0:
                time.sleep(t)
        else:
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    detectface2()