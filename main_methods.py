import cv2
import numpy as np
from utils import *
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy import stats
import os
from types import SimpleNamespace
'''
使用灰度重心算法获得激光坐标
主要用于验证原理,如果需要提升精度也可以考虑steger算法
'''
def getLaserCo(img,roi_height,roi_width,threshold):
    #首先进行roi裁剪
    roi = img[roi_height,roi_width]
    #提取红色通道,根据不同需要,现场中也可以考虑2R-B-R的方法
    red = roi[:,:,2]
    #获得激光区域,并且抹平threshold
    laser = np.where(red>threshold,red-threshold,0)
    #使用灰度重心法获得激光中心坐标
    centers = []
    rows,cols = laser.shape
    for col in range(cols):
        y_value = laser[:,col]
        y_index = np.where(y_value>0)[0]
        if(len(y_index)>0):
            weights = y_value[y_index].astype(float)
            fenzi = np.sum(y_index * weights)
            fenmu = np.sum(weights)
            center_y = fenzi/fenmu
            #这里将坐标还原为图像绝对坐标,方便后续处理
            centers.append((col+roi_width.start,center_y+roi_height.start))
    centers= np.array(centers)
    
        # --- 基于斜率剔除反光点 ---
    if len(centers) > 10:
        # 1. 计算每个点的局部斜率
        slopes = []
        for i in range(1, len(centers)-1):
            dx = centers[i+1, 0] - centers[i-1, 0]
            if dx != 0:
                slope = (centers[i+1, 1] - centers[i-1, 1]) / dx
            else:
                slope = 0
            slopes.append(abs(slope))
        
        slopes = np.array(slopes)
        # 2. 找出斜率异常大的点（反光区域通常斜率突变）
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        outlier_indices = []
        
        for i in range(1, len(centers)-1):
            if abs(slopes[i-1]) > mean_slope + 2 * std_slope:
                outlier_indices.append(i)
        
        # 3. 剔除异常点
        mask = np.ones(len(centers), dtype=bool)
        mask[outlier_indices] = False
        centers = centers[mask]
    return centers

'''提取激光线时的测试函数,以便在不改变主函数的情况下提升激光提取效果'''
def getLaserCo_Test(img, roi_height, roi_width, threshold):
    # 首先进行roi裁剪
    roi = img[roi_height, roi_width]
    # 提取红色通道
    red = roi[:, :, 2]
    # 获得激光区域
    laser = np.where(red > threshold, red - threshold, 0)
    
    # 灰度重心法
    centers = []
    rows, cols = laser.shape
    for col in range(cols):
        y_value = laser[:, col]
        y_index = np.where(y_value > 0)[0]
        if len(y_index) > 0:
            weights = y_value[y_index].astype(float)
            fenzi = np.sum(y_index * weights)
            fenmu = np.sum(weights)
            center_y = fenzi / fenmu
            centers.append((col + roi_width.start, center_y + roi_height.start))
    
    centers = np.array(centers)
    
    # --- 新增：基于斜率剔除反光点 ---
    if len(centers) > 10:
        # 1. 计算每个点的局部斜率
        slopes = []
        for i in range(1, len(centers)-1):
            dx = centers[i+1, 0] - centers[i-1, 0]
            if dx != 0:
                slope = (centers[i+1, 1] - centers[i-1, 1]) / dx
            else:
                slope = 0
            slopes.append(abs(slope))
        
        slopes = np.array(slopes)
        # 2. 找出斜率异常大的点（反光区域通常斜率突变）
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        outlier_indices = []
        
        for i in range(1, len(centers)-1):
            if abs(slopes[i-1]) > mean_slope + 2 * std_slope:
                outlier_indices.append(i)
        
        # 3. 剔除异常点
        mask = np.ones(len(centers), dtype=bool)
        mask[outlier_indices] = False
        centers = centers[mask]
    
    return centers

'''
使用RANSAC方法拟合交点
'''    
def getIntersectPoint(centers):
    
    intersect_x, intersect_y = None, None
    
    if centers is None or len(centers) < 10: 
        return None, None
    try:
        X = centers[:,0].reshape(-1,1)
        y = centers[:,1]
        model = LinearRegression()
        r0 = RANSACRegressor(estimator=model,min_samples=5,residual_threshold=1,max_trials=50,random_state=42)
        r0.fit(X,y)
        k0 = r0.estimator_.coef_[0]
        b0 = r0.estimator_.intercept_
        inner_mask = r0.inlier_mask_

        '''
        对第二条线进行ransec拟合,但不要直接使用第二条线上的点来画图
        '''

        line2_points = centers[~inner_mask]        
        #如果直接取反的话,那么line2_points上多了许多1不要的点,所以需要 对 line2进行再次拟合.
        
        r1 = RANSACRegressor(estimator=model,min_samples=2,residual_threshold=1.5,max_trials=100,random_state=42)
        r1.fit(line2_points[:,0].reshape(-1,1),line2_points[:,1])
        k1 = r1.estimator_.coef_[0]
        b1 = r1.estimator_.intercept_

        #在画第二条线的时候,不要依赖于内点,直接根据斜率和截距即可.
        if abs(k0 - k1) > 0.05: # 确保两条线不平行
            intersect_x = (b1 - b0) / (k0 - k1)
            intersect_y = k0 * intersect_x + b0
        
        # 绘制唯一的中心点
        # cv2.circle(display2, (int(intersect_x), int(intersect_y)), 7, [255, 255, 255], 2)
    except Exception as e:
        print(f"拟合失败: {e}")
    return intersect_x,intersect_y

'''
使用kf滤波平滑交点,并且计算交点的高度
'''
class LaserTracker:
    def __init__(self,base_lutPath=None,k=None,b=None,mtx=None,dist=None,base_distannce=None):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1     # 过程噪声
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5 # 测量噪声
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100        # 初始估计误差
        self.is_initialized = False
        #丢失计数器,用于统计丢失的步数
        self.miss_count = 0
        self.lut = None
        if base_lutPath and os.path.exists(base_lutPath):
            try:
                self.lut = np.load(base_lutPath)
                print("成功加载 LUT 基准文件")
            except Exception as e:
                print(f"警告：读取 LUT 文件失败: {e}")
        else:
            print("提示：未找到 LUT 文件，将跳过高度计算")
        self.k = k
        self.b = b
        self.mtx = mtx
        self.dist = dist
        #相机光心到物体的距离
        self.base_distance = base_distannce
    def getHeight(self,x,y):
        if self.lut is None:
            return 0
        idx  = int(round(x))
        if 0<idx<len(self.lut): 
            basey = self.lut[idx]
            if basey !=0:
                return (y - basey)*self.k + self.b
        return 0
    
    def get_physical_coords(self,rect_x,rect_y):
        height = self.getHeight(rect_x,rect_y)
        z_actual = self.base_distance - height
        fx = self.mtx[0,0]
        fy = self.mtx[1,1]
        cx = self.mtx[0,2]
        x = (rect_x-cx)*z_actual/fx
        return x
    def update(self,measure_x,measure_y):

        if measure_x is None or measure_y is None or np.isnan(measure_x):
            self.miss_count += 1
            if self.miss_count>30:
                #如果丢失30步,重置初始化状态
                self.is_initialized = False
            #只预测,不更新 
            prediction = self.kf.predict()
            return prediction[0,0],prediction[1,0]   
        #对预测进行初始化
        if not self.is_initialized:
            initial_state = np.array([
                [measure_x],
                [measure_y],
                [0],
                [0]
            ],dtype=np.float32)
            self.kf.statePre = initial_state
            self.kf.statePost = initial_state
            self.is_initialized = True
            return measure_x, measure_y
        self.miss_count = 0
        self.kf.predict()
        measurement = np.array([[np.float32(measure_x)], [np.float32(measure_y)]], dtype=np.float32)
        update_state = self.kf.correct(measurement=measurement)
        return update_state[0,0],update_state[1,0]


'''动态roi窗格'''       
def get_dynamic_roi(center_x, center_y, window_w=200, window_h=150, img_shape=(1080, 1920)):
    """
    根据交点坐标，生成一个新的矩形 ROI 区域
    """
    # 计算边界，并确保不超出图像范围
    if center_x is not None:
        x_start = max(0, int(center_x - window_w // 2))
        x_stop = min(img_shape[1], int(center_x + window_w // 2))
        
        y_start = max(0, int(center_y - window_h // 2))
        y_stop = min(img_shape[0], int(center_y + window_h // 2))
        roi_h,roi_w = slice(y_start, y_stop), slice(x_start, x_stop)
    else:
        # roi_h,roi_w = slice(50+100,200+100),slice(750+50,1000-50)
        roi_h,roi_w = slice(100,1080),slice(500,1500)
    return roi_h,roi_w

'''动态roi窗格2'''       
# 在 utils.py 中修改：
def get_dynamic_roi2(x, y, window_w=200, window_h=150, img_shape=(1080, 1920)):
    """返回 SimpleNamespace 格式的 ROI"""
    # if x is None or y is None:
    #     return SimpleNamespace(
    #         h=slice(100, 1080),
    #         w=slice(500, 1500)
    #     )
    
    x_start = max(0, int(x - window_w // 2))
    x_stop = min(img_shape[1], int(x + window_w // 2))
    y_start = max(0, int(y - window_h // 2))
    y_stop = min(img_shape[0], int(y + window_h // 2))
    
    return SimpleNamespace(
        h=slice(y_start, y_stop),
        w=slice(x_start, x_stop)
    )
    


'''用于对摄像头宽高进行初始化,宽度1920,高度1080'''

def capInit(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

'''获得激光线中心的坐标后,将之画出来'''
def drawLaserCenters(centers,img):
    for (x,y) in centers:
        cv2.circle(img,(int(x),int(y)),2,[0,255,0],-1)
        
        
'''
使用平均滤波方式进行平滑处理
'''
class CoordinateSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history_x = []
        self.history_y = []

    def update(self, x, y):

        if x is None or y is None:
            if len(self.history_x) > 0:
                return int(self.history_x[-1]), int(self.history_y[-1])
            return None, None

        # 添加新坐标
        self.history_x.append(x)
        self.history_y.append(y)

        if len(self.history_x) > self.window_size:
            self.history_x.pop(0)
            self.history_y.append(y) # 修正补齐逻辑
            self.history_y.pop(0)
        avg_x = int(np.mean(self.history_x))
        avg_y = int(np.mean(self.history_y))
        return avg_x, avg_y

'''使用动态roi'''

  


'''
使用线性差值法补全一下为0的数据
'''
def interpValues(arr):
    res = arr.copy()
    nans = (res == 0)
    # 如果全为0，直接返回，防止 np.interp 报错
    if not np.any(~nans):
        return res
    x_vals = np.where(~nans)[0]
    y_vals = res[~nans]
    res[nans] = np.interp(np.where(nans)[0], x_vals, y_vals)
    return res

'''
坐标转换
'''


"""将像素坐标还原为去畸变后的坐标"""
def get_pt_rect(u, v, mtx, dist):
    src_pt = np.array([[[u, v]]], dtype=np.float32)
    dst_pt = cv2.undistortPoints(src_pt, mtx, dist, P=mtx)
    x, y = dst_pt.ravel()
    return x,y



__all__ =[
    'capInit',
    'drawLaserCenters',
    'listToDict',
    'getLaserCo',
    'getIntersectPoint',
    'CoordinateSmoother',
    'LaserTracker',
    'interpValues',
    'get_pt_rect',
    'getLaserCo_Test',
    'get_dynamic_roi',
    'get_dynamic_roi2'

]