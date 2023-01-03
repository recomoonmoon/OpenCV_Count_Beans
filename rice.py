import cv2 as cv
import numpy as np
import os
import matplotlib as plt

images_path = "./rice_images"
names = os.listdir(images_path)
images = [cv.imread(i) for i in [os.path.join(images_path, j) for j in os.listdir(images_path)]]
#读入图片，并存储在列表images中
kernel = np.ones((7, 7), np.uint8)
for i in range(len(images)):
    num = int(names[i].split(".")[0])
    img = images[i]
    cv.imshow('原图', img)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #从彩色转为灰度图
    cv.imshow('灰度', gray_img)
    ret, mask = cv.threshold(gray_img, 120, 255, cv.THRESH_BINARY) #二值化
    cv.imshow('二值化', mask)
    erosion = cv.erode(mask, kernel, iterations=1)  # 腐蚀变换
    cv.imshow('腐蚀化', erosion)
    dist_img = cv.distanceTransform(erosion, cv.DIST_L1, cv.DIST_MASK_3)  # 距离变换
    cv.imshow('距离变换', dist_img)
    dist_output = cv.normalize(dist_img, 0, 1.0, cv.NORM_MINMAX)
    ret, mask2 = cv.threshold(dist_output * 80, 0.3, 255, cv.THRESH_BINARY)
    cv.imshow('饱满化', mask2)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)
    opening = np.array(opening, np.uint8)
    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 轮廓提取，一个轮廓就是一粒豆子
    key = cv.waitKey(0)
    cv.waitKey(0)
    print(f"第{i+1}副图片")
    print(f'共有豆子：{num} 粒  预测豆子：{len(contours)} 粒')
    print(f"误差为: {abs(len(contours) - num)}")
    print()

