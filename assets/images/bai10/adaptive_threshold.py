import numpy as np
import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread('coin.jpg',0)

bins_num = 256
hist, bin_edges = np.histogram(gray_img, bins=bins_num)
weight1 = np.cumsum(hist)
weight2 = np.cumsum(hist[::-1])[::-1]
print(weight1.shape)
print(weight2)
