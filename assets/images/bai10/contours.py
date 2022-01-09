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
draw_img = cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)

print(type(contours),len(contours))
# for c in contours:
#     # find bounding box coordinates
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
#     # find minimum area
#     rect = cv2.minAreaRect(c)
    
#     # calculate coordinates of the minimum area rectangle
#     box = cv2.boxPoints(rect)
    
#     # normalize coordinates to integers
#     box = np.int0(box)
    
#     # draw contours
#     cv2.drawContours(img, [box], 0, (0,0, 255), 3)
    
#     # calculate center and radius of minimum enclosing circle
#     (x, y), radius = cv2.minEnclosingCircle(c)
#     # cast to integers
#     center = (int(x), int(y))
#     radius = int(radius)
#     # draw the circle
#     rgb_img = cv2.circle(img, center, radius, (255, 255, 0), 2)
# # cv2.drawContours(rgb_img, contours, -1, (255, 0, 0), 2)
# cv2.imshow("contours", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

    
fig = plt.figure(figsize=(16, 4))

fig.add_subplot(151)
plt.title('Original'),plt.axis('off'),plt.imshow(rgb_img)

fig.add_subplot(152)
plt.title('Edge'),plt.axis('off'),plt.imshow(edge)

fig.add_subplot(153)
plt.title('Contours'),plt.axis('off'),plt.imshow(draw_img)

plt.show()