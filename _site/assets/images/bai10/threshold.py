import numpy as np
import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread('coin.jpg',0)

T = 125 # threshold value

_, thresh_img = cv2.threshold(gray_img,T,255,cv2.THRESH_BINARY)
_, thresh_img_inv = cv2.threshold(gray_img,T,255,cv2.THRESH_BINARY_INV)
X = cv2.bitwise_and(gray_img, thresh_img_inv)

plt.figure(figsize=(16, 4))
plt.subplot(151),plt.imshow(gray_img,cmap='gray'),plt.title('Origin'),plt.axis(False)
plt.subplot(152),plt.imshow(thresh_img,cmap='gray'),plt.title('THRESH BINARY'),plt.axis(False)
plt.subplot(153),plt.imshow(thresh_img_inv,cmap='gray'),plt.title('THRESH BINARY INV'),plt.axis(False)
plt.subplot(154),plt.imshow(X,cmap='gray'),plt.title('Bitwise AND'),plt.axis(False)
plt.show()