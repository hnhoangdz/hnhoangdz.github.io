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
edge = cv2.Canny(blurred,30,150)

# Finds contours
contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # find bounding box coordinates
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    
    # draw the circle
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)
    
draw_img = cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)
    
fig = plt.figure(figsize=(16, 4))

fig.add_subplot(151)
plt.title('Original'),plt.axis('off'),plt.imshow(rgb_img)

fig.add_subplot(152)
plt.title('Contours'),plt.axis('off'),plt.imshow(draw_img)

fig.add_subplot(153)
plt.title('Other shape'),plt.axis('off'),plt.imshow(img)

plt.show()