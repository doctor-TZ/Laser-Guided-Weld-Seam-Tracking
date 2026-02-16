import pytest
import numpy as np
import cv2
from main_methods import getLaserCo, getIntersectPoint, LaserTracker, interpValues

# --- 1. Fixtures: 准备模拟数据 ---

@pytest.fixture
def synthetic_laser_img():
    """构造一张带有红色激光线的黑底图"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # 在第 100 行画一条红线 (BGR格式: [0, 0, 255])
    img[100, 50:150, 2] = 255 
    return img

@pytest.fixture
def v_shape_centers():
    """构造一个 V 字型的中心点集合，用于测试交点拟合"""
    # 线1: y = x (0到10)
    line1 = np.array([[i, i] for i in range(20)])
    # 线2: y = -x + 20 (11到20)
    line2 = np.array([[i, -i + 40] for i in range(21, 40)])
    return np.vstack((line1, line2))

# --- 2. Unit Tests ---

def test_getLaserCo_basic(synthetic_laser_img):
    # Arrange
    roi_h = slice(80, 120)
    roi_w = slice(40, 160)
    threshold = 100
    
    # Act
    centers = getLaserCo(synthetic_laser_img, roi_h, roi_w, threshold)
    
    # Assert
    assert len(centers) > 0
    # 验证提取的 Y 坐标是否接近我们画线的 100 像素位置
    # (注意：centers 返回的是绝对坐标)
    assert 99 <= centers[0][1] <= 101

def test_getIntersectPoint_accuracy(v_shape_centers):
    # Arrange (v_shape_centers 是 y=x 和 y=-x+40 的点)
    # 预期交点在 (20, 20)
    
    # Act
    ix, iy = getIntersectPoint(v_shape_centers)
    
    # Assert
    assert ix is not None
    assert 19 <= ix <= 21
    assert 19 <= iy <= 21

def test_laser_tracker_init(tmp_path):
    # Arrange
    # 构造一个空的 LUT 文件
    lut_file = tmp_path / "test_lut.npy"
    np.save(lut_file, np.linspace(0, 100, 1920))
    
    # Act
    tracker = LaserTracker(base_lutPath=str(lut_file), k=1.0, b=0.0, 
                           mtx=np.eye(3), base_distannce=500.0)
    
    # Assert
    assert tracker.is_initialized is False
    assert tracker.lut is not None

def test_laser_tracker_update_and_miss_logic():
    # Arrange
    tracker = LaserTracker(k=1.0, b=0.0, mtx=np.eye(3), base_distannce=500.0)
    
    # Act 1: 第一次更新，应该完成初始化
    x1, y1 = tracker.update(100, 100)
    assert tracker.is_initialized is True
    
    # Act 2: 模拟连续丢失
    for _ in range(31):
        tracker.update(None, None)
    
    # Assert: 丢失超过30步，应该重置
    assert tracker.is_initialized is False

def test_interpValues():
    # Arrange: 构造中间有0的序列
    arr = np.array([1.0, 0.0, 0.0, 4.0])
    
    # Act
    result = interpValues(arr)
    
    # Assert: 0.0 应该被插值为 2.0 和 3.0
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(result, expected)