import numpy as np
import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread('text.jpg',0)
blurred = cv2.GaussianBlur(gray_img,(5,5),0)

# Apply threhold
_, thresh_img = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY)
_, otsu_img = cv2.threshold(blurred, 0, 255, \
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)

fig = plt.figure(figsize=(12, 10))

fig.add_subplot(221)
plt.title('Original')
plt.set_cmap('gray')
plt.axis('off')
plt.imshow(gray_img)

fig.add_subplot(222)
plt.title('T = 125')
plt.axis('off')
plt.imshow(thresh_img)
plt.set_cmap('gray')

fig.add_subplot(223)
plt.title('Otsu')
plt.axis('off')
plt.set_cmap('gray')
plt.imshow(otsu_img)

fig.add_subplot(224)
plt.title('Adp')
plt.imshow(thresh)
plt.axis('off')
plt.set_cmap('gray')

plt.show()