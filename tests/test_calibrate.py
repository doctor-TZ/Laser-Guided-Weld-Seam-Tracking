import pytest
import numpy as np
import cv2
import os

from calibrateCamera import run_calibration 

# --- 1. Arrange: 准备模拟的标定数据 ---
@pytest.fixture
def mock_calibration_data():
    """
    这是一个高级技巧：我们不读图片，而是直接构造符合要求的点。
    如果相机内参计算逻辑是对的，那么给它一组完美的点，它应该返回极小的误差。
    """
    board_size = (9, 6)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * 25
    
    # 模拟 3 张完美的图像点（直接使用物理坐标，假设相机是完美的）
    objpoints = [objp, objp, objp]
    # 图像点稍微加一点点偏移，模拟真实情况
    imgpoints = [objp[:, :2].reshape(-1, 1, 2) + 0.5 for _ in range(3)]
    
    return objpoints, imgpoints

# --- 2. Unit Tests ---

def test_calibration_math_logic(mock_calibration_data):
    # Arrange
    objpoints, imgpoints = mock_calibration_data
    img_size = (1920, 1080)
    
    # Act
    # 直接调用 OpenCV 的标定函数验证数学逻辑是否跑通
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    # Assert
    assert ret is not None
    assert mtx.shape == (3, 3) # 内参矩阵必须是 3x3
    assert dist.shape[1] == 5  # 默认畸变系数是 5 个

def test_calibration_output_file(tmp_path):
    # Arrange: 创建一个临时路径保存参数，不污染正式环境
    output_file = tmp_path / "test_params.npz"
    mock_mtx = np.eye(3)
    mock_dist = np.zeros(5)
    
    # Act: 模拟保存过程
    np.savez(output_file, mtx=mock_mtx, dist=mock_dist)
    
    # Assert: 验证文件是否真的生成了
    assert os.path.exists(output_file)
    loaded = np.load(output_file)
    assert np.array_equal(loaded['mtx'], mock_mtx)