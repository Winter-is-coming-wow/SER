# -- coding: utf-8 --
import cv2 as cv
import numpy as np
def pre_process(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img=cv.resize(img,(640,480))
    img_mean=np.array([127,127,127])
    img=(img-img_mean)/128
    #cv.imshow('img',img)
    cv.waitKey(0)
    img=np.transpose(img,[2,0,1])

    img=np.expand_dims(img,0)
    img=img.astype(np.float32)

if __name__ == '__main__':
    filename='cache/1.jpg'
    img=cv.imread(filename)
    pre_process(img)