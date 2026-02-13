import cv2
import numpy as np
import open3d as o3d


def to_gray(bgr_img):
    """将 BGR 图像转换为灰度图"""
    if bgr_img is None:
        raise ValueError("输入图像不能为空")
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)


def to_hsv(bgr_img):
    """将 BGR 图像转换为 HSV 色彩空间"""
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)


def to_lab(bgr_img):
    """将 BGR 图像转换为 LAB 色彩空间"""
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)


def to_binary(gray_img, thresh):
    """普通固定阈值二值化"""
    if gray_img is None:
        raise ValueError("灰度图像不能为空")
    _, binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    return binary


def to_adaptive_binary_mean(gray_img, block_size=11, c=2,invert=False):
    """
    自适应均值阈值二值化
    block_size: 必须为奇数且 >= 3，建议 3~51 之间
    """
    if gray_img is None:
        raise ValueError("灰度图像不能为空")
    
    if block_size % 2 == 0:
        block_size += 1
        # print(f"block_size 自动调整为奇数: {block_size}")  # 建议注释掉或改用 logging
    
    if block_size < 3:
        block_size = 3
        
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        threshold_type,
        block_size, c
    )

def draw_text(img, text, pos):
    """
    自用简化版：默认绿色、小号、细字体
    """
    cv2.putText(img, str(text), pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 255, 0], 2)

def showPoints(points): 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    

def get_rect_kernel(size=5):
    """创建矩形形态学核"""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def get_ellipse_kernel(size=5):
    """创建椭圆形态学核"""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def get_cross_kernel(size=5):
    """创建十字形形态学核"""
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))



def show(title,img):
    cv2.imshow(title,img)

def resize(img,targetWidth):
    h,w = img.shape[:2]
    ratio = h/w
    targetHeight = int(targetWidth*ratio)
    resized_img = cv2.resize(img,(targetWidth,targetHeight),interpolation=cv2.INTER_AREA)
    return resized_img

def findExtrContours(img):
    contours,hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours,hier

def findTreeContours(img):
    contours,hier = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours,hier

def area(cnt):
    return cv2.contourArea(cnt)

def generate_uniform(low,high,size):
    return np.random.uniform(low,high,size)

def listToDict(alist):
    return {int(x):y for x,y in alist} 

__all__ = [
    'to_gray',
    'to_hsv',
    'to_lab',
    'to_binary',
    'to_adaptive_binary_mean',
    'get_rect_kernel',
    'get_ellipse_kernel',
    'get_cross_kernel',
    'show',
    'resize',
    'findExtrContours',
    'findTreeContours',
    'area',
    'generate_uniform',
    'showPoints',
    'draw_text',
    'listToDict'
]