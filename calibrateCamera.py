import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import glob

def run_calibration(image_paths, board_size=(9, 6), square_size=25):

    CHECKERBOARD = (9,6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((54, 3), np.float32)

    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objp = objp * 25

    objpoints = []
    imgpoints = []

    imgs = glob.glob('../Images/chessboard/*.jpg')

    imgs.sort()

    for path in imgs:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        # flags=0 或者是特殊的预处理模式
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp) # 对应的物理坐标是统一的
            
            # 亚像素精确化（让坐标从整数变成更准的浮点数，比如 100.25）
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # # 可视化：画出来看看找得准不准
            # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # cv2.imshow('Calibration Finding...', img)
            # cv2.waitKey(100) # 每张图停 100 毫秒
            
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("标定成功！")
        print("\n重投影误差ret:\n",ret)
        print("\n相机内参矩阵 (Camera Matrix):\n", mtx)
        print("\n畸变系数 (Distortion Coefficients):\n", dist)
        
        # 保存结果，以后做激光扫描直接加载，不用重复标定
        np.savez("camera_params.npz", mtx=mtx, dist=dist)
    return ret, mtx, dist