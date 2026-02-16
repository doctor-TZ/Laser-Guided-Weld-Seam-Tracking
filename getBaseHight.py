import cv2
import numpy as np
from main_methods import *
from utils import *
import pandas as pd


'''
本程序用于获取激光平面基础高度数据
use this program to test base height
'''
ROI_H = slice(10,540)
ROI_W = slice(300,1100)

cap = cv2.VideoCapture(0)
capInit(cap=cap)
#使用查找表方式存储高度数据
base_lut = np.zeros(1920, dtype=np.float32)
while True:
    ret,frame = cap.read()
    if not ret:
        print('摄像头工作不正常')
        break
     #绘制辅助框
    cv2.rectangle(frame,(ROI_W.start,ROI_H.start),(ROI_W.stop,ROI_H.stop),[100,100,100],1)
    
    centers = getLaserCo(frame,roi_height=ROI_H,roi_width=ROI_W,threshold=200)
    drawLaserCenters(centers,frame)
    key = cv2.waitKey(1)&0xff
    if(key==ord('d')):
        for x,y in centers:
            idx = int(round(x))
            if 0<idx<1920:
                base_lut[idx] = y
        #使用线性差值法补全一下0区域
        base_lut = interpValues(base_lut)        
        np.save('data/base_lut.npy',base_lut)
        print('基础高度已提取')
    show('frame',frame)
    if(key==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()