import pytest
import cv2
import numpy as np
import utils



@pytest.fixture
def sample_bgr():
    """创建一个 100x200 (高x宽) 的三通道测试图像"""
    return np.zeros((100, 200, 3), dtype=np.uint8)

@pytest.fixture
def sample_gray():
    """创建一个 100x200 的单通道测试图像"""
    return np.zeros((100, 200), dtype=np.uint8)

# --- Unit Tests ---
def test_to_gray_success(sample_bgr):
    # 1. Arrange (由 fixture 提供 sample_bgr)
    
    # 2. Act
    result = utils.to_gray(sample_bgr)
    # 3. Assert
    assert result.shape == (100, 200)
    assert len(result.shape) == 2  # 确认变为单通道
    

def test_to_gray_none_input():
    # 1. Arrange
    invalid_input = None
    # 2. Act & 3. Assert
    with pytest.raises(ValueError, match="输入图像不能为空"):
        utils.to_gray(invalid_input)


def test_resize_keeps_ratio(sample_bgr):
    # 1. Arrange
    # 原始 100x200 (高x宽)，比例 H/W = 0.5
    target_width = 400
    # 2. Act
    resized_img = utils.resize(sample_bgr, target_width)
    # 3. Assert
    # 预期高度 = 400 * 0.5 = 200
    assert resized_img.shape[1] == 400
    assert resized_img.shape[0] == 200
    
def test_to_adaptive_binary_mean_fixes_block_size(sample_gray):
    # 1. Arrange
    # 输入偶数 block_size，函数内部应自动处理
    even_block_size = 10
    # 2. Act
    result = utils.to_adaptive_binary_mean(sample_gray, block_size=even_block_size)
    
    # 3. Assert
    assert result is not None
    assert result.shape == sample_gray.shape

def test_get_rect_kernel():
    # 1. Arrange
    size = 7
    
    # 2. Act
    kernel = utils.get_rect_kernel(size)
    
    # 3. Assert
    assert kernel.shape == (7, 7)
    assert np.all(kernel == 1) # 矩形核内部应全是1


def test_listToDict():
    # 1. Arrange
    raw_list = [(10.5, "laser"), (20.0, "weld")]
    
    # 2. Act
    result = utils.listToDict(raw_list)
    
    # 3. Assert
    # 验证 key 是否被转成了 int，内容是否匹配
    assert result == {10: "laser", 20: "weld"}
    assert isinstance(list(result.keys())[0], int)

def test_generate_uniform():
    # 1. Arrange
    low, high, size = 0, 10, (5, 5)
    
    # 2. Act
    data = utils.generate_uniform(low, high, size)
    
    # 3. Assert
    assert data.shape == (5, 5)
    assert np.all(data >= 0) and np.all(data <= 10)