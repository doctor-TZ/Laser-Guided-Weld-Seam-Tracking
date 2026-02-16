# tests/conftest.py
import pytest
import numpy as np
import cv2

@pytest.fixture(scope="session")
def camera_params():
    """全项目共享：模拟一套相机内参矩阵"""
    mtx = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    return {"mtx": mtx, "dist": dist}

@pytest.fixture
def black_img():
    """全项目共享：创建一个 1080p 的黑底图"""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)