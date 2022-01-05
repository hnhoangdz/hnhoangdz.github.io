import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('coin2.jpg')
rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = rgb_img.copy() 

# Blur image by GaussianBlur
blurred = cv2.GaussianBlur(gray_img,(11,11),0)

# Edge detection by Canny -> binary image
edge = cv2.Canny(blurred,30,180)

# Finds contours
contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
draw_img = cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)
print(hierarchy)
fig = plt.figure(figsize=(16, 4))

fig.add_subplot(151)
plt.title('Original'),plt.axis('off'),plt.imshow(rgb_img)

fig.add_subplot(152)
plt.title('Edge'),plt.axis('off'),plt.imshow(edge)

fig.add_subplot(153)
plt.title('Contours'),plt.axis('off'),plt.imshow(draw_img)

plt.show()