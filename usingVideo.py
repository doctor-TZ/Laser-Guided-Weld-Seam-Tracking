import cv2
import numpy as np
from main_methods import *
from utils import *
import pandas as pd
import time
import traceback
import json
import os


if __name__=='__main__':
    # --- 配置区域 ---
    #获取lut信息 
    base_lut_path = 'data/base_lut.npy'
    #获取相机的内参与畸变系数
    with np.load('data/camera_params.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
    BASE_DISTANCE = 550     #mm
    #获取三角测量系数
    config_file = 'data/config.json'
    if os.path.exists(config_file):
        with open(config_file,'r')  as f:
            config = json.load(f)
            K_TRI = config.get('K_TRI')
            B_TRI = config.get('B_TRI')
    #当前ROI
    ROI_H = slice(50+100,200+100)
    ROI_W = slice(750+50,1000-50)


    
    # 记录数据用的列表
    history_data = []
    
    
    #从视频中读取
    path = 'Images/hanfeng/output.mp4'
    cap = cv2.VideoCapture(path)
    capInit(cap=cap)
    #当前帧,用于控制后续用于加速播放
    current_frame = 0
    kf = LaserTracker(base_lutPath=base_lut_path,k=K_TRI,b=B_TRI,mtx=mtx,dist=dist)

    prev_time = time.time()
    
    try:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                print('视频帧已读取完毕')
                break
            #计算FPS
            now = time.time()
            fps = 1/(now - prev_time)
            prev_time = now
            
            img = frame.copy()
            x = None
            y = None
            height = None
            #绘制辅助框
            cv2.rectangle(img,(ROI_W.start,ROI_H.start),(ROI_W.stop,ROI_H.stop),[100,100,100],1)

            #算法核心
            centers = getLaserCo(img,roi_height=ROI_H,roi_width=ROI_W,threshold=200)
            
            drawLaserCenters(centers=centers,img=img)
            
            intersect_x,intersect_y= getIntersectPoint(centers=centers)

            
            
            #当有交点的时候,焊枪点火,没有交点的时候,焊枪熄火
            if (intersect_x is None or intersect_y is None):
                print('当前无焊点')
                draw_text(img,'No welding point',(100,300))
                
                x, y = None, None 
                ROI_H, ROI_W = get_dynamic_roi(None, None)
            
            else:
                #去除畸变
                intersect_x, intersect_y = get_pt_rect(intersect_x, intersect_y, mtx, dist)
            
                #滤波与测量
                x,y = kf.update(intersect_x,intersect_y)
                #绘制动态roi窗格
                ROI_H,ROI_W = get_dynamic_roi(intersect_x,intersect_y)
                height =kf.getHeight(x,y)
            
                #增强可视化
                cv2.circle(img,(int(x),int(y)),5,[0,0,255],-1)
                draw_text(img,f'height:{height:.2f}mm frame:{int(fps)}',(int(x)-80,int(y)-80))
                draw_text(img,f'center:{x:.2f} y:{y:.2f}',(int(x)-80,int(y)-100))
            #记录数据
            data = [time.time(),x,y,height] 
            history_data.append(data)
            
            show('img',img)
            key = cv2.waitKey(1)&0xff
            if (key==ord('q')):
                break
            elif key == ord('n'):  # 前进10秒
                current_frame += int(10 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == ord('p'):  # 后退10秒
                current_frame -= int(10 * fps)
                current_frame = max(0, current_frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)        
    except Exception as e:
        print('程序运行异常')
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # 退出时保存数据
        if history_data:
            df = pd.DataFrame(history_data, columns=['time', 'x', 'y', 'height'])
            df.to_csv('data/weld_log.csv', index=False)
            print("焊接轨迹数据已保存")





