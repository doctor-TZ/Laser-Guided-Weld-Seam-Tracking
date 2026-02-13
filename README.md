# RoboWeld-Vision: 低成本激光视觉焊缝实时追踪系统
# RoboWeld-Vision: A Real-Time Laser Seam Tracking System with Sub-pixel Precision

在线演示地址:
https://www.bilibili.com/video/BV1zpc5zoEx4/
视频截图:
<img width="1519" height="881" alt="Snipaste_2026-02-13_10-05-28" src="https://github.com/user-attachments/assets/78badd96-636c-44e0-91fe-5aa1dffd91dd" />

## 为什么做这个项目? | Why this project?
   
**此项目作为应聘 CV 算法工程师的展示 Demo，复刻了工业级线结构光扫描器的核心 Pipeline。虽然硬件配置属于消费级，但在软件算法层面实现了完整的抗干扰策略与亚像素级提取算法。 本项目不仅展示了计算机视觉在工业自动化中的实际应用，更体现了在有限硬件预算下追求极致鲁棒性的工程思维。**

**This project serves as a technical demonstration for a CV Algorithm Engineer position, replicating the core pipeline of industrial-grade laser stripe scanners. While utilizing consumer-grade hardware, the system implements a complete suite of anti-interference strategies and sub-pixel precision validation at the algorithmic level. It represents my understanding of robustness, real-time performance, and cost-effective mass-production solutions in industrial vision. Beyond demonstrating the practical application of Computer Vision in factory automation, this project reflects an engineering mindset: pursuing ultimate robustness under limited hardware budgets.**

## 核心原理 | Main Implementation


### 1. 深度感知 | Depth Perception
**基于单目线结构光三角测量模型，将像素偏移量映射为物理深度：**

$$\Delta Z = \frac{f \cdot B}{d \cdot \cos(\theta)}$$

**Based on the laser triangulation model, mapping pixel displacement to physical depth:**


### 2. 特征点识别与鲁棒性 | Seam Localization & Robustness
对于 V 型/L 型焊缝，激光线的交点即为焊缝位置。我们利用**灰度重心法**提取亚像素中心线，并通过算法求取交点。结合**卡尔曼滤波 (Kalman Filter)**，显著改善了系统的鲁棒性，有效避免了焊缝点的跳动或环境干扰。
$$L_1: y = k_1x + b_1, \quad L_2: y = k_2x + b_2 \implies P_{seam} = L_1 \cap L_2$$
** For V-shaped or L-shaped seams, the intersection of the laser lines marks the precise welding point. We use the **Gray-Center-of-Gravity** method for sub-pixel extraction and calculate the intersection. Integrated with **Kalman Filtering**, the system's robustness is significantly enhanced, effectively suppressing jitter and environmental interference.

## 🔍 算法细节 | Technical Principles

### 1. 亚像素条纹中心提取 | Sub-pixel Stripe Extraction
> **实现平衡实时性与精度的核心。**

** 为了在嵌入式端（如树莓派）实现高帧率，本项目采用 **灰度重心法 (Gray-Center-of-Gravity)**。通过对激光条纹横截面的灰度分布进行加权计算，突破像素分辨率限制，实现亚像素级定位。
** To achieve high frame rates on embedded devices (e.g., Raspberry Pi), this project implements the **Gray-Center-of-Gravity** method, calculating the intensity-weighted centroid to surpass pixel resolution limits:

$$P_{sub} = \frac{\sum_{i=1}^{n} (i \cdot G_i)}{\sum_{i=1}^{n} G_i}$$



---

### 2. 镜面反射抑制 | Specular Reflection Suppression
> **解决金属反光导致的伪影问题。**

** 激光照射在金属表面会产生反光或光晕。在进行 **RANSAC** 线段拟合前，系统先对激光中心点坐标进行预筛选，剔除斜率异常的噪点。这一预处理确保了输入拟合模型的数据更加纯净，从而极大提升了焊缝定位的准确性。
** Laser radiation on metallic surfaces often induces specular reflections or halos. Prior to RANSAC line fitting, the system performs a pre-filtering process on the laser coordinates to exclude outliers with abnormal slopes. This ensures a cleaner dataset for the fitting model, leading to significantly higher precision in seam localization.



---

### 3. 卡尔曼滤波状态估算 | Kalman Filter Smoothing
> **提升轨迹的平滑度与抗干扰能力。**

** 系统采用 **卡尔曼滤波 (Kalman Filter)** 对计算出的交点坐标进行平滑处理。该机制有效减少了输出坐标的异常抖动，并增强了系统对抗弧光、飞溅等环境噪声干扰的能力。
** A **Kalman Filter** is implemented to smooth the calculated intersection coordinates. This mechanism effectively suppresses stochastic jitter in the output data and enhances the system's resilience against environmental interference (e.g., arc light or welding splashes).



---

### 4. 动态感兴趣区域 | Dynamic ROI (Region of Interest)
> **兼顾算法鲁棒性与计算效率。**
** 系统根据焊缝位置实时更新 **动态搜索窗格 (ROI)**。这不仅大幅降低了图像处理的计算开销，还通过排除无关区域的干扰，进一步提升了算法在复杂背景下的鲁棒性。
** The system generates a **Dynamic ROI** based on the real-time seam position. This not only significantly reduces computational overhead but also enhances algorithmic robustness by isolating the search space from peripheral background noise.

---

## 🚀 性能表现 | Performance (Experimental Results)

| 指标 (Metric) | 表现 (Performance) | 备注 (Notes) |
| :--- | :--- | :--- |
| **FPS (Processing Speed)** | 15 - 30 Hz | 在消费级 PC/嵌入式设备实测 (On standard PC hardware) |
| **Static Error (Absolute)** | $\pm 0.5$ mm | 受限于简易激光器线宽 (Limited by laser line width) |
| **Success Rate (Recognition)** | > 90% | 能有效应对金属表面反光干扰 (Robust against reflections) |


## 硬件搭建| Hardware Setup
   本项目采用普通廉价的线激光发生器与 USB 摄像头。安装时需注意相机与激光平面的夹角应保持在 $45^\circ$ 至 $60^\circ$ 之间，以确保三角测量模型具有最佳的深度灵敏度。
   The system is built using a cost-effective line laser module and a standard USB camera. During assembly, the mounting angle between the camera and the laser plane must be maintained at approximately $45^\circ$ to $60^\circ$. This specific geometric configuration ensures optimal depth sensitivity for the triangulation model.
![IMG_2108](https://github.com/user-attachments/assets/4fcf47fe-ff55-4ce1-9127-d04621403684)






   
## 如何运行 | How to run?
1. 克隆项目: `git clone https://github.com/用户名/项目名.git`
2. 安装依赖: `pip install -r requirements.txt`
3. 运行程序:
   - 视频版: `python usingVideo.py`
   - 摄像头版: `python usingCamera.py` (需要自己搭建硬件,关键是你你需要固定住摄像头和激光器,然后二者的夹角在45-60°/Hardware setup is required. The key is to securely fix the camera and the laser line generator, maintaining an inclusion angle between $45^\circ$ and $60^\circ$.)


---

## 📄 版权声明 | Copyright

**本项目代码仅作为个人简历展示及技术交流使用，未经许可严禁任何形式的商用、转载或二次分发。** **All Rights Reserved.** The source code in this repository is for personal portfolio demonstration and technical exchange only. No part of this project may be copied, modified, or distributed for commercial purposes without explicit permission.

