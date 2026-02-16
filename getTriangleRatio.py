import cv2
import numpy as np
from utils import *
from main_methods import *
import glob
import json

'''
使用最小二乘法拟合偏移量与高度的数据
use this program to get the ratio between the height and offset, and save it to json
'''

roi_height = slice(900,1050)
roi_width = slice(650,800)



'''
使用垫片获得不同高度的偏移量

z0:0
z1:7.6
z2:10.2
z3:17.8
z4:20.4
z5:30.2
z6:40.4
'''

centers_dicts = []

imgs = glob.glob('Images/deapthImgs/*.jpg')

imgs.sort()

ROI_H = slice(10,410)
ROI_W = slice(730,1100)

for path in imgs:
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图片: {path}")
        continue
    centers = getLaserCo(img,roi_height=ROI_H,roi_width=ROI_W,threshold=200)
    drawLaserCenters(img=img,centers=centers)
    temp_dict = {int(x): y for x, y in centers}
    centers_dicts.append(temp_dict)
    show('img',img)
    if cv2.waitKey(0) & 0xff == ord('q'):
        break
    

# 假设 imgs[0] 是 z0 (基准面)
z0_dict = centers_dicts[0]
heights = [0, 5.5, 13, 56.7] # 你的物理高度数据

avg_offsets = []

for i, zi_dict in enumerate(centers_dicts):
    offsets = []
    # 只有在 z0 中也存在的 x 坐标才计算差值
    for x, y_val in zi_dict.items():
        if x in z0_dict:
            offsets.append(y_val - z0_dict[x])
    
    if offsets:
        res = np.mean(offsets)
        avg_offsets.append(res)
        print(f"图片 {i} (高度 {heights[i]}mm) 的平均像素偏移: {res:.4f}")
    else:
        avg_offsets.append(0)

# 最终拟合高度与偏移的关系
# np.polyfit(像素偏移, 物理高度, 1) -> 得到系数 K
k_coeffs = np.polyfit(avg_offsets, heights, 1)
print(f"\n最终转换公式: Height = {k_coeffs[0]:.4f} * delta_y + {k_coeffs[1]:.4f}")

#保存线性结果与偏移量之间的斜率关系
k_val = float(k_coeffs[0])
b_val = float(k_coeffs[1])

config ={
    "K_TRI":k_val,
    "B_TRI":b_val,
    "description":'焊缝追踪项目中的高度标定数据'
}

with open('data/config.json','w') as f:
    json.dump(config,f,indent=4)
print('配置已保存到JSON')