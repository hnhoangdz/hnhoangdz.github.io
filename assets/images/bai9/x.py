import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test.png')

small_k = cv2.medianBlur(image, 3)
medium_k = cv2.medianBlur(image, 5)
big_k = cv2.medianBlur(image, 7)
plt.figure(figsize=(16, 5))
plt.subplot(151),plt.imshow(image),plt.title('Origin Image'),plt.axis(False)
plt.subplot(152),plt.imshow(small_k),plt.title('k = 3'),plt.axis(False)
plt.subplot(153),plt.imshow(medium_k),plt.title('k = 5'),plt.axis(False)
plt.subplot(154),plt.imshow(big_k),plt.title('k = 7'),plt.axis(False)
plt.show()